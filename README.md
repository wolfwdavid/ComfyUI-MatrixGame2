# ComfyUI-MatrixGame2

[Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) (Skywork, MIT) as
ComfyUI nodes: an image plus keyboard and camera actions becomes first-person video, a block of 12
frames per action, and the world stays resident on the server so the next prompt continues it.

Built for the Comfy developer platform (serverless workers, one `comfy deploy run` per move), where
flash-attn is not available: the vendored model code in `mg2core/` runs every attention through
`torch.nn.functional.scaled_dot_product_attention`. No compiled extensions, no CUDA-specific wheels.

## Nodes

**Matrix-Game 2.0 Loader** (`MatrixGame2Loader`)
Picks the three weights from the ComfyUI model folders and keeps them loaded:

| folder | file (from [Skywork/Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0)) |
|---|---|
| `models/diffusion_models/` | `base_distilled_model/base_distill.safetensors` (6.5 GB, universal model, 3 denoising steps) |
| `models/vae/` | `Wan2.1_VAE.pth` |
| `models/clip_vision/` | `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (only the vision tower is used) |

**Matrix-Game 2.0 Walk** (`MatrixGame2Walk`)
Inputs: `model`, `image` (any size, center-cropped and resized to 640x352), `actions` (comma
separated, one per block), `session` (world id), `seed`. Outputs: `frames` as an IMAGE batch
(12 fps; the first block of a world gives 9 frames, later blocks 12) and `session_info` (JSON). The
node also writes the same JSON next to the outputs so an API client gets the world state back.

Actions: `forward`, `back`, `left`, `right`, `camera_l`, `camera_r`, `camera_up`, `camera_down`,
diagonals `camera_ul` ... `camera_dr`, combinations joined with `_` such as `forward_left` or
`forward_camera_r`, and `none`.

Sessions: pass the `session` string from an earlier run to continue that world (the KV caches,
the conditioning and the streaming VAE decoder cache are kept in memory; the last 3 worlds stay).
If the id is unknown, for instance after the worker restarted, a new world starts from `image`, so
always send the latest frame together with the id. A world holds 120 latent frames (40 blocks).

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/wolfwdavid/ComfyUI-MatrixGame2
pip install -r ComfyUI-MatrixGame2/requirements.txt
```

Needs a CUDA GPU, roughly 16 GB free by estimate (bf16 transformer 3.6 GB, CLIP, VAE, plus ~1.7 GB of
caches per world). CPU runs work for development with a reduced config but are slow.

## Status

- Verified on CPU: the full pipeline with random weights and a 1-layer config (first block 9 frames,
  next block 12 frames, session resume, world-full error, restart), and node registration plus graph
  validation in ComfyUI 0.37.
- Built into a Comfy developer platform release (torch 2.12 + CUDA 13, ComfyUI 0.37, RTX PRO 6000 MIG
  slice): the pack and its requirements install cleanly on the worker.
- Not yet run on a GPU with the real weights, so speed per block and image quality are unmeasured.

## What is vendored and changed

`mg2core/` holds the files of the official `Matrix-Game-2` folder that inference needs (`wan/modules`,
`wan/vae/wanx_vae_src`, `demo_utils/vae_block3.py`, `utils/scheduler.py`, the universal config). Changes:

- `attention.py` (both copies) replaced by an SDPA implementation with the flash-attn signatures;
  `action_module.py` imports `flash_attn_func` from it instead of flash-attn.
- The import-time `torch.compile(flex_attention)` is dropped; the KV-cache path never calls it.
- Rotary tables pick CUDA when available instead of asserting it; the text tokenizer import is lazy.
- `runner.py` is the streaming pipeline (`CausalInferenceStreamingPipeline.inference`) as a
  `Session` object: conditioning and caches are built once per world, `step(actions)` runs one block
  per action with the same denoising schedule, clean context pass and chunked VAE decode.

License: MIT (this repo), Matrix-Game 2.0 MIT, Wan2.1 code Apache-2.0.
