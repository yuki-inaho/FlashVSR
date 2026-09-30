#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiled 4x streaming inference for high-resolution videos on limited VRAM.

The streaming feature cache of FlashVSR grows with the target resolution, so a
full-frame 4x run of a 720p video (5120x2816) can exceed a 48 GB GPU. This
runner splits each frame into 2x2 overlapping tiles, runs the official
long-video streaming pipeline per tile, then blends the seams and writes a
single H.264 mp4.

Usage:
    python infer_flashvsr_v1.1_tiny_long_video_tiled.py <video-or-folder> [more...]
"""

import os, sys, importlib.util
import numpy as np
from PIL import Image
import imageio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "flashvsr_long_base",
        os.path.join(HERE, "infer_flashvsr_v1.1_tiny_long_video.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base()

OVERLAP_W = 128
OVERLAP_H = 128


def compute_layout(h0, w0, scale):
    s = int(round(scale))
    tile_w = ((w0 + OVERLAP_W + 63) // 2) // 32 * 32
    tile_h = ((h0 + OVERLAP_H + 63) // 2) // 32 * 32
    if tile_w <= 0 or tile_h <= 0 or tile_w >= w0 or tile_h >= h0:
        return None
    tW_tile, tH_tile = tile_w * s, tile_h * s
    if tW_tile % 128 or tH_tile % 128:
        return None
    col_starts = [0, w0 - tile_w]
    row_starts = [0, h0 - tile_h]
    tW_full = (w0 * s // 128) * 128
    tH_full = (h0 * s // 128) * 128
    union_w = col_starts[1] * s + tW_tile
    union_h = row_starts[1] * s + tH_tile
    crop_w = union_w - tW_full
    crop_h = union_h - tH_full
    if crop_w < 0 or crop_h < 0 or crop_w % 2 or crop_h % 2:
        return None
    return dict(
        s=s, tile_w=tile_w, tile_h=tile_h, tW_tile=tW_tile, tH_tile=tH_tile,
        col_starts=col_starts, row_starts=row_starts,
        tW_full=tW_full, tH_full=tH_full,
        left_crop=crop_w // 2, top_crop=crop_h // 2,
    )


class LazyTile:
    """Lazily crops + upscales a rectangular source region."""

    def __init__(self, src_frames, idx, scale, y0, y1, x0, x1, tW, tH,
                 dtype=torch.bfloat16, cache_size=12):
        self.src_frames = src_frames
        self.idx = idx
        self.scale = scale
        self.y0, self.y1, self.x0, self.x1 = y0, y1, x0, x1
        self.tW, self.tH = tW, tH
        self.dtype = dtype
        self.cache_size = cache_size
        self._cache = {}

    def _get_frame(self, i):
        key = self.idx[i]
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        arr = self.src_frames[key][self.y0:self.y1, self.x0:self.x1]
        img = Image.fromarray(arr).convert("RGB")
        img = img.resize((self.tW, self.tH), Image.BICUBIC)
        t = base.pil_to_tensor_neg1_1(img, self.dtype, "cpu")
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = t
        return t

    def __getitem__(self, key):
        frame_slice = key[2]
        frames = [self._get_frame(i) for i in range(frame_slice.start, frame_slice.stop)]
        return torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)


def chunk_to_uint8(chunk):
    frames = chunk[0].permute(1, 2, 3, 0).float()
    frames = ((frames + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return frames.numpy()


class TileCollector:
    def __init__(self, path, shape):
        self.path = path
        self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
        self.n = 0

    def __call__(self, chunk):
        frames = chunk_to_uint8(chunk)
        self.mm[self.n:self.n + len(frames)] = frames
        self.n += len(frames)

    def close(self, expected):
        self.mm.flush()
        del self.mm
        assert self.n == expected, f"{self.path}: wrote {self.n}, expected {expected}"


def main():
    inputs = sys.argv[1:] or ["./inputs/example4.mp4"]
    RESULT_ROOT = "./results"
    os.makedirs(RESULT_ROOT, exist_ok=True)

    seed, scale, dtype = 0, 4.0, torch.bfloat16
    sparse_ratio = 2.0
    local_range = 11

    pipe = base.init_pipeline()

    for path in inputs:
        name = os.path.basename(path.rstrip("/"))
        if name.startswith("."):
            continue

        rdr = imageio.get_reader(path)
        meta = {}
        try:
            meta = rdr.get_meta_data()
        except Exception:
            pass
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        src_frames = []
        try:
            n = 0
            while True:
                try:
                    src_frames.append(rdr.get_data(n))
                except Exception:
                    break
                n += 1
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        total = len(src_frames)
        if total <= 0:
            print(f"[Error] no frames in {name}")
            continue
        h0, w0 = src_frames[0].shape[:2]
        print(f"[{name}] Original Resolution: {w0}x{h0} | Frames: {total} | FPS: {fps}")

        layout = compute_layout(h0, w0, scale)
        if layout is None:
            print(f"[Error] {name}: cannot tile this resolution; use the non-tiled script.")
            continue

        idx = list(range(total)) + [total - 1] * 4
        F = base.largest_8n1_leq(len(idx))
        idx = idx[:F]
        out_frames = F - 4
        print(f"[{name}] Target Frames (8n-3): {out_frames}")

        s = layout["s"]
        tW_full, tH_full = layout["tW_full"], layout["tH_full"]
        print(f"[{name}] Full target: {tW_full}x{tH_full} (2x2 tiles of "
              f"{layout['tile_w']}x{layout['tile_h']} src)")

        call_kwargs = dict(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            num_frames=F, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (tH_full * tW_full),
            kv_ratio=3.0,
            local_range=local_range,
            color_fix=True,
        )

        tile_files = {}
        for ri, y0 in enumerate(layout["row_starts"]):
            for ci, x0 in enumerate(layout["col_starts"]):
                tag = f"{ri}{ci}"
                y1, x1 = y0 + layout["tile_h"], x0 + layout["tile_w"]
                tmp = os.path.join(RESULT_ROOT, f".tile{tag}_{name.split('.')[0]}.npy")
                print(f"[{name}] Tile {tag}: src y={y0}:{y1} x={x0}:{x1} "
                      f"-> {layout['tW_tile']}x{layout['tH_tile']}")
                collector = TileCollector(
                    tmp, (out_frames, layout["tH_tile"], layout["tW_tile"], 3))
                LQ = LazyTile(src_frames, idx, scale, y0, y1, x0, x1,
                              layout["tW_tile"], layout["tH_tile"])
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
                pipe(LQ_video=LQ, height=layout["tH_tile"], width=layout["tW_tile"],
                     frame_callback=collector, **call_kwargs)
                collector.close(out_frames)
                tile_files[tag] = tmp
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()

        print(f"[{name}] Blending and encoding H.264...")
        tiles = {tag: np.load(p, mmap_mode="r") for tag, p in tile_files.items()}
        x_start = layout["col_starts"][1] * s
        blend_w = layout["tW_tile"] - x_start
        blend_h = layout["tH_tile"] - layout["row_starts"][1] * s
        alpha_h = np.linspace(0.0, 1.0, blend_w, dtype=np.float32)[None, :, None]
        alpha_v = np.linspace(0.0, 1.0, blend_h, dtype=np.float32)[:, None, None]
        tc, lc = layout["top_crop"], layout["left_crop"]

        save_path = os.path.join(
            RESULT_ROOT, f"FlashVSR_v1.1_Tiny_Long_{name.split('.')[0]}_seed{seed}.mp4")
        writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=5)
        try:
            for fi in range(out_frames):
                union_w = layout["tW_full"] + lc * 2
                top = np.empty((layout["tH_tile"], union_w, 3), np.uint8)
                bot = np.empty_like(top)
                top[:, :x_start] = tiles["00"][fi][:, :x_start]
                left = tiles["00"][fi][:, x_start:x_start + blend_w].astype(np.float32)
                right = tiles["01"][fi][:, :blend_w].astype(np.float32)
                top[:, x_start:x_start + blend_w] = ((1 - alpha_h) * left + alpha_h * right).astype(np.uint8)
                top[:, x_start + blend_w:] = tiles["01"][fi][:, blend_w:]

                bot[:, :x_start] = tiles["10"][fi][:, :x_start]
                left = tiles["10"][fi][:, x_start:x_start + blend_w].astype(np.float32)
                right = tiles["11"][fi][:, :blend_w].astype(np.float32)
                bot[:, x_start:x_start + blend_w] = ((1 - alpha_h) * left + alpha_h * right).astype(np.uint8)
                bot[:, x_start + blend_w:] = tiles["11"][fi][:, blend_w:]

                if lc:
                    top = top[:, lc:lc + tW_full]
                    bot = bot[:, lc:lc + tW_full]

                out = np.empty((tH_full, tW_full, 3), np.uint8)
                n_top = layout["row_starts"][1] * s - tc
                out[:n_top] = top[tc:tc + n_top]
                lo = tc + n_top
                left_v = top[lo:lo + blend_h].astype(np.float32)
                right_v = bot[:blend_h].astype(np.float32)
                out[n_top:n_top + blend_h] = ((1 - alpha_v) * left_v + alpha_v * right_v).astype(np.uint8)
                out[n_top + blend_h:] = bot[blend_h:blend_h + (tH_full - n_top - blend_h)]

                writer.append_data(out)
        finally:
            writer.close()
        del tiles
        for p in tile_files.values():
            os.remove(p)

        print(f"[{name}] Saved: {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
