# FlashVSR オンボーディング

> 新しい開発者・LLMエージェントが、このリポジトリを最短で安全に動かすための入口です。
> コマンドは特記がない限りリポジトリルート、推論コマンドは `examples/WanVSR` で実行します。

**更新:** 2026-09-30
**対象:** `yuki-inaho/FlashVSR`（upstream: `OpenImagingLab/FlashVSR`）
**目的:** uv 環境とビルド不要の Block-Sparse-Attention で FlashVSR v1.1 を実行し、高解像度・長尺動画も現実的なVRAM/RAMで処理できるようにする。

## 1. 最初に行うこと

1. `README.md` と本書を読む。
2. `git status --short --branch` で既存変更を確認する。
3. `uv sync` で Python 3.11 環境を構築する。
4. Block-Sparse-Attention のビルド済みホイールを導入する（自前ビルド不要）。
5. Hugging Face から v1.1 重みを `examples/WanVSR/FlashVSR-v1.1/` へ取得する。
6. `examples/WanVSR` で短い example 動画を推論し、出力 mp4 を `ffprobe` で検査する。
7. 長尺・高解像度は `LOW_MEM=1` またはタイル版スクリプトを使う。
8. commit 前に、重み・結果・個人データ・絶対パスが staged diff に無いことを確認する。

## 2. プロジェクト概要

FlashVSR は one-step streaming の diffusion ベース動画超解像（VSR）実装で、4x 超解像を得意とし、低品質動画の復元（デブラー相当）にも使える。推論は `diffsynth/` のパイプラインが担当する。

```text
LQ動画 (128の倍数へ crop)
  -> LQ_proj_in (Causal_LQ4x_Proj)
  -> Wan DiT + Locality-Constrained Sparse Attention (block_sparse_attn)
  -> TCDecoder (v1.1 tiny) または Wan2.1 VAE (full)
  -> H.264 mp4
```

- `FlashVSRTinyPipeline`: TCDecoder による高速構成（推奨）。
- `FlashVSRTinyLongPipeline`: streaming で長尺に対応。`frame_callback` による逐次書き出しに対応。
- `FlashVSRFullPipeline`: Wan2.1 VAE を使う構成（`Wan2.1_VAE.pth` が必要）。
- Sparse attention は `block_sparse_attn` に依存する。**LCSA を外すと品質が落ちる**ため、必ず導入する。

## 3. 主要なファイルと責務

| パス | 用途 |
| :--- | :--- |
| `README.md` | upstream の説明とセットアップ |
| `docs/ONBOARDING.md` | 本書 |
| `pyproject.toml` / `uv.lock` | uv 用の依存定義（torch cu124、BSA 以外の実行依存） |
| `.python-version` | Python 3.11.13 固定 |
| `requirements.txt` / `setup.py` | upstream 由来。`pyproject.toml` があるため通常は不要 |
| `diffsynth/` | 推論ライブラリ本体（vendored） |
| `diffsynth/models/wan_video_dit.py` | DiT / sparse attention。`FLASHVSR_FFN_CHUNK` で FFN を分割 |
| `diffsynth/pipelines/flashvsr_tiny_long.py` | streaming パイプライン。`frame_callback` で逐次出力 |
| `examples/WanVSR/infer_flashvsr_v1.1_tiny.py` | 短尺アプリ（`sys.argv` で入力差し替え可） |
| `examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py` | 長尺アプリ。`LOW_MEM=1` で低メモリ動作 |
| `examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video_tiled.py` | 2x2 タイル推論（高解像度・低VRAM向け） |
| `examples/WanVSR/utils/`, `prompt_tensor/` | TCDecoder 定義、固定 prompt tensor |
| `examples/WanVSR/inputs/` | 同梱サンプル動画（tracked） |
| `examples/WanVSR/FlashVSR-v1.1/` | 取得した重み。**Git管理外** |
| `examples/WanVSR/results/` | 出力動画と一時 `.npy`。**Git管理外** |

## 4. 前提条件

- `uv`（Python 3.11.13 を自動取得）
- NVIDIA GPU（CUDA 12.4 以上が動くドライバ）。A6000 48GB で検証済み
- ディスク: 環境 + 重みで約 10GB 以上。タイル推論時は一時 `.npy` が数GB〜30GB 程度
- 実行は `examples/WanVSR` を cwd にすること（重み・prompt・入力が相対パスで解決される）

目安:

| 構成 | VRAM 目安 | 備考 |
| :--- | :--- | :--- |
| 480p以下 4x（tiny/long） | 16〜24GB | `LOW_MEM=1` で RAM も小さく |
| 720p 4x 全体（5120x2816級） | 48GB超 | KVキャッシュが解像度²で増えるため単一GPUでは不可 |
| 720p 4x タイル（2x2） | 48GBで動作 | シームはブレンドして1本のmp4に再構築 |

## 5. セットアップ

### 5.1 uv 環境

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 5.2 Block-Sparse-Attention（ビルド不要）

配布ホイールは GitHub Release にある。インストーラが torch / CUDA / ABI を自動判定する。

```bash
# pip 環境
curl -fsSL https://github.com/yuki-inaho/FlashVSR/releases/download/bsa-wheels-v0.0.2/install_bsa.sh | bash

# uv 環境
curl -fsSL https://github.com/yuki-inaho/FlashVSR/releases/download/bsa-wheels-v0.0.2/install_bsa.sh | bash -s -- --uv
```

| ホイール | torch / CUDA | 対応GPU |
| :--- | :--- | :--- |
| `...+cu12torch2.6cxx11abifalse...` | 2.6.0 + cu124 | sm_80/90: RTX 30/40, A100, A6000, H100（sm86/89 は sm80 cubin） |
| `...+cu12torch2.8cxx11abitrue...` | 2.8.0 + cu128 | sm_80/90/100/120: 上記 + RTX 50, B200 |

- ホイールは **torch のマイナーバージョンと Python 3.11 に一致**させること。
- import 確認は必ず torch を先に読み込む（`libc10.so` は torch が提供する）。

```bash
uv run python -c "import torch; import block_sparse_attn; print('BSA OK')"
```

### 5.3 モデル重み

```bash
cd examples/WanVSR
uv run hf download JunhaoZhuang/FlashVSR-v1.1 --local-dir ./FlashVSR-v1.1
```

`diffusion_pytorch_model_streaming_dmd.safetensors`, `LQ_proj_in.ckpt`, `TCDecoder.ckpt`, `Wan2.1_VAE.pth` などが配置される。full 構成では `Wan2.1_VAE.pth` が必要。重みは Git 管理外。

HF トークンや認証情報はコマンドライン・文書・ログ・commit message に書かない。

## 6. 標準ワークフロー

すべて `examples/WanVSR` で実行。出力は `results/*.mp4`（H.264 / libx264 / yuv420p）。

### 6.1 短尺クリップ

```bash
uv run python infer_flashvsr_v1.1_tiny.py                # 同梱 example 4本
uv run python infer_flashvsr_v1.1_tiny.py /path/<clip>.mp4
```

### 6.2 長尺（streaming）

```bash
uv run python infer_flashvsr_v1.1_tiny_long_video.py /path/<clip>.mp4
```

RAM が少ない環境（コンテナの cgroup 制限など）では逐次処理にする。

```bash
LOW_MEM=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python infer_flashvsr_v1.1_tiny_long_video.py /path/<clip>.mp4
```

`LOW_MEM=1` は「4x に拡大した全フレームを RAM に保持しない（遅延拡大）」+「出力を逐次 H.264 書き出し」に切り替える。

### 6.3 高解像度 / 低VRAM（2x2 タイル）

720p の 4x（5120x2816 級）は、streaming の KV キャッシュが解像度に比例して増え、48GB GPU では単一フレームでも収まらない。タイル版は 2x2 のオーバーラップタイルで推論し、境界を線形ブレンドして 1 本の mp4 に再構築する。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FLASHVSR_FFN_CHUNK=16384 \
uv run python infer_flashvsr_v1.1_tiny_long_video_tiled.py /path/<clip>.mp4
```

- タイル寸法は入力の向き（回転メタデータ適用後の縦横）から自動計算し、目標解像度が 128 の倍数になるよう調整する。
- 一時ファイルは `results/.tile*.npy`。処理後に削除される（失敗時は手動削除）。
- 出力名は `results/FlashVSR_v1.1_Tiny_Long_<name>_seed0.mp4`。

### 6.4 full 構成（Wan2.1 VAE）

```bash
uv run python infer_flashvsr_v1.1_full.py
# または infer_flashvsr_v1.1_tiny_long_video_tiled.py 等は tiny 系のみ
```

## 7. メモリ・性能の契約

- 出力の目標解像度は **128 の倍数**（`compute_scaled_and_target_dims`）。4x 設定を推奨。
- KV キャッシュ量は概ね **解像度に比例**（1ブロックあたり `kv_ratio` ステップ分）。`kv_ratio=3.0` が既定。
- 入力動画に回転メタデータがある場合、デコード後のフレーム配列は表示向き（縦横がメタデータと入れ替わる）になる。スクリプトはデコード後の shape から寸法を決めるので、そのままでよい。
- 生成動画は音声を持たない（upstream と同じ）。

環境変数:

| 変数 | 用途 |
| :--- | :--- |
| `LOW_MEM=1` | long 版で遅延拡大 + 逐次 H.264 書き出し |
| `FLASHVSR_FFN_CHUNK=<N>` | DiT の FFN を N トークンずつ計算（VRAM 削減、結果は同一） |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | CUDA 断片化対策 |

推論パラメータ（upstream 既定に準拠）:

| パラメータ | 推奨 | 意味 |
| :--- | :--- | :--- |
| `sparse_ratio` / `topk_ratio` | 1.5 or 2.0 | 1.5 速い / 2.0 安定 |
| `local_range` | 9 or 11 | 9 シャープ / 11 安定 |
| `kv_ratio` | 3.0 | 過去ステップ参照量。下げると省メモリだが時間的一貫性が落ちる |

## 8. 検証

```bash
# 出力の契約
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_frames,pix_fmt,duration \
  -of default=noprint_wrappers=1 results/<output>.mp4
```

- H.264（`codec_name=h264`, `pix_fmt=yuv420p`）で保存されることを確認する。
- フレーム数は入力より数フレーム少なくなる（8n-3 へ調整と末尾パディング除去のため）。
- 品質確認の例: 入力と出力の Laplacian 分散を比較する（尺度依存のため、bicubic 4x と比較するとわかりやすい）。

BSA ホイールの導入検証は `import torch` → `import block_sparse_attn` の順で行う。余力があれば `block_sparse_attn_func` を小さなテンソルで実行し、SDPA と比較する。

## 9. 公開・秘密情報の境界

Git へ含めるもの:

- source code、docs、`pyproject.toml`、`uv.lock`、`.python-version`、`.gitignore`
- 同梱サンプル（`inputs/`、`prompt_tensor/`）

Git へ含めないもの（`.gitignore` 済み）:

- モデル重み（`examples/WanVSR/FlashVSR*/`、`*.pth`、`*.ckpt`、`*.safetensors`）
- 出力（`examples/WanVSR/results/`、`*.mp4` の生成物、`*.npy` の一時タイル）
- ログ、`outputs/`、`tmp/`、`temp/`
- HF トークン等の認証情報、`.env`
- 個人の動画・データのファイル名や絶対パス、作業セッションのログ

commit 前に stage した差分を検査する:

```bash
git status --short
git diff --cached --stat
git diff --cached --no-ext-diff --unified=0 | \
  grep '^+' | grep -v '^+++' | \
  grep -E '(/home/|/Users/|HF_TOKEN=|hf_[A-Za-z0-9]{20,})'
```

ヒットした場合は内容を表示せず unstage し、相対パスやプレースホルダに置き換える。

## 10. トラブルシューティング

### `libc10.so: cannot open shared object file`

`block_sparse_attn` を torch より先に import している。`import torch` を先に行う（`diffsynth` 経由では問題にならない）。

### CUDA OOM

- タイル版を使う。次に `kv_ratio` を下げる。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付ける。
- 他プロセスの GPU 使用量（`nvidia-smi`）を確認する。
- 解像度を下げるのは最終手段（4x 学習モデルのため品質が落ちる）。

### RAM OOM（プロセスが `Killed`）

コンテナの cgroup 制限（`/sys/fs/cgroup/memory.max`）を確認する。long 版は全フレームを保持する経路があるため、`LOW_MEM=1` かタイル版を使う。

### `No wan_video_vae models available`

tiny 系では正常（VAE 不要）。full 系でのみ `Wan2.1_VAE.pth` を配置する。

### ホイールの不一致

`torch のマイナーバージョン` と `Python 3.11` が一致するホイールを選ぶ。release に無い組み合わせは自前ビルドする（下記）。

### Block-Sparse-Attention を自前ビルドする場合

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
git submodule update --init --recursive
CUDA_HOME=/usr/local/cuda-12.6 \
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE \
BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90" \
MAX_JOBS=6 NVCC_THREADS=2 \
uv pip install --no-build-isolation .
```

- `BLOCK_SPARSE_ATTN_CUDA_ARCHS` が解釈するのは `80;90;100;110;120`。86/89 は sm80 cubin で動く。
- ビルドはメモリを大量に使う。cgroup 制限下では `MAX_JOBS` / `NVCC_THREADS` を下げる。
- ビルド分離環境の setuptools が新しすぎると `pkg_resources` が無く失敗する。`setuptools<81` を使う。
- Blackwell 向けは CUDA 12.8+ と torch 2.8+cu128 の組み合わせでビルドする。

## 11. オンボーディング完了チェックリスト

- [ ] `README.md` と本書を読んだ
- [ ] `git status --short --branch` で作業ツリーを確認した
- [ ] `uv sync` が成功した
- [ ] BSA ホイールを導入し、`import block_sparse_attn` が成功した
- [ ] v1.1 重みを `examples/WanVSR/FlashVSR-v1.1/` に取得した
- [ ] サンプル動画の推論が成功し、`ffprobe` で H.264 を確認した
- [ ] 高解像度はタイル版、低RAMは `LOW_MEM=1` を使うと理解した
- [ ] 重み・結果・一時ファイル・秘密情報を stage していない
- [ ] `git diff --cached` に個人パスやトークンが無いことを確認した

## 12. 更新履歴

- 2026-09-30: 初版。uv 環境、BSA ビルド済みホイール（cu124/cu128）、v1.1 推論、低メモリ・タイル推論、公開境界を記載。
