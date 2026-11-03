"""ComfyUI nodes: load Matrix-Game 2.0 once, then walk a world by keyboard/camera actions."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import folder_paths

from . import runner

try:
    import comfy.model_management as mm
except Exception:  # pragma: no cover - outside ComfyUI
    mm = None


def _device():
    if mm is not None:
        return mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatrixGame2Loader:
    """Loads the distilled universal transformer, the Wan2.1 VAE and the CLIP vision tower from
    models/diffusion_models, models/vae and models/clip_vision. The result stays resident."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transformer": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "base_distill.safetensors from Skywork/Matrix-Game-2.0"}),
            "vae": (folder_paths.get_filename_list("vae"), {"tooltip": "Wan2.1_VAE.pth from the same repo"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"tooltip": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}),
            "compile_decoder": ("BOOLEAN", {"default": False, "tooltip": "torch.compile the streaming VAE decoder (minutes of warm-up, faster blocks)"}),
        }}

    RETURN_TYPES = ("MG2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MatrixGame2"

    def load(self, transformer, vae, clip_vision, compile_decoder):
        dev = _device()
        if mm is not None:
            mm.soft_empty_cache()
        m = runner.get_models(folder_paths.get_full_path_or_raise("diffusion_models", transformer),
                              folder_paths.get_full_path_or_raise("vae", vae),
                              folder_paths.get_full_path_or_raise("clip_vision", clip_vision),
                              dev, compile_decoder=compile_decoder)
        return (m,)


class MatrixGame2Walk:
    """Advance a world. `session` names a world kept in memory by an earlier prompt on this server;
    when it is unknown (first move, or the server restarted) a new world starts from `image`, so a
    client should always send the latest frame together with the session id. One action per block
    of 12 frames (9 for the first block); actions are comma separated: forward, back, left, right,
    camera_l, camera_r, camera_up, camera_down, combos like forward_left or forward_camera_r, none."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MG2_MODEL",),
            "image": ("IMAGE", {"tooltip": "start frame; any size, center-cropped to 640x352"}),
            "actions": ("STRING", {"default": "forward, forward, forward, forward", "multiline": True}),
            "session": ("STRING", {"default": "", "tooltip": "world id to continue; empty = new world with a fresh id"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "filename_prefix": ("STRING", {"default": "mg2"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "session_info")
    FUNCTION = "walk"
    CATEGORY = "MatrixGame2"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")

    def walk(self, model: runner.Models, image: torch.Tensor, actions: str, session: str, seed: int, filename_prefix: str):
        acts = [a.strip() for a in actions.replace("\n", ",").split(",") if a.strip()]
        if not acts:
            raise ValueError("no actions given")
        frame = (image[0].detach().cpu().float().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        frame = runner.fit_frame(frame)
        t0 = time.time()
        sid = session.strip()
        s = runner.get_session(sid) if sid else None
        resumed = s is not None and s.m is model and s.remaining_blocks >= len(acts)
        if not resumed:
            s = runner.new_session(model, frame, int(seed) % (2 ** 63), sid=sid or None)
        t_init = time.time() - t0
        frames = s.step(acts)
        info = s.info()
        info.update(resumed=resumed, seconds=round(time.time() - t0, 2), init_seconds=round(t_init, 2),
                    actions=acts, frames=int(frames.shape[0]))
        out_dir = folder_paths.get_output_directory()
        full, name, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, frames.shape[2], frames.shape[1])
        fn = f"{name}_{counter:05}_.json"
        with open(os.path.join(full, fn), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        images = torch.from_numpy(frames.astype(np.float32) / 255.0)
        return {"ui": {"images": [{"filename": fn, "subfolder": sub, "type": "output"}], "text": [json.dumps(info)]},
                "result": (images, json.dumps(info))}


NODE_CLASS_MAPPINGS = {"MatrixGame2Loader": MatrixGame2Loader, "MatrixGame2Walk": MatrixGame2Walk}
NODE_DISPLAY_NAME_MAPPINGS = {"MatrixGame2Loader": "Matrix-Game 2.0 Loader", "MatrixGame2Walk": "Matrix-Game 2.0 Walk"}
