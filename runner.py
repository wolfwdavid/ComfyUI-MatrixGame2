"""
runner.py -- Matrix-Game 2.0 (Skywork, MIT) as a resident engine: load the weights once, keep
interactive "worlds" (sessions) alive between ComfyUI prompts, and advance a world by a list of
actions, one action per block of 3 latent frames (= 12 pixel frames, 9 for the first block).

This is the KV-cache streaming path of the official pipeline/causal_inference.py rewritten so that
the state lives in a Session object instead of local variables of one blocking loop:
  * Session.__init__  == the image conditioning + cache allocation of inference()
  * Session.step()    == one iteration of the "temporal denoising loop", repeated per action
Everything numerical (shift, denoising steps, cond_current slicing, context pass, streaming VAE
decode) follows the official code; see mg2core/ for the vendored modules.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from .mg2core.constant import ZERO_VAE_CACHE
from .mg2core.scheduler import FlowMatchScheduler
from .mg2core.vae_block3 import VAEDecoderWrapper
from .mg2core.vae_src.clip import clip_xlm_roberta_vit_h_14
from .mg2core.vae_src.vae import WanVAE
from .mg2core.wan_modules.causal_model import CausalWanModel

HERE = os.path.dirname(os.path.abspath(__file__))
H, W = 352, 640                 # the model's native frame
LH, LW = 44, 80                 # latent grid
FRAME_SEQ = (LH // 2) * (LW // 2)   # 880 tokens per latent frame
BLOCK = 3                       # latent frames per action
FPS = 12                        # what the official export writes
DENOISE = [1000, 666, 333]      # universal distilled model
SHIFT = 5.0
CONTEXT_NOISE = 0

KEYS = {"forward": 0, "back": 1, "left": 2, "right": 3}
CAM = 0.1
CAMERA = {  # order matters: two-letter diagonals before their one-letter prefixes
    "camera_ul": [CAM, -CAM], "camera_ur": [CAM, CAM], "camera_dl": [-CAM, -CAM], "camera_dr": [-CAM, CAM],
    "camera_up": [CAM, 0], "camera_down": [-CAM, 0], "camera_l": [0, -CAM], "camera_r": [0, CAM],
}
WORD_ALIASES = {"look_left": "camera_l", "look_right": "camera_r", "look_up": "camera_up", "look_down": "camera_down",
                "camera_left": "camera_l", "camera_right": "camera_r"}
TOKEN_ALIASES = {"w": "forward", "s": "back", "a": "left", "d": "right",
                 "none": "", "stay": "", "idle": "", "wait": "", "q": ""}


def parse_action(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """'forward', 'forward_left', 'forward_camera_r', 'camera_l', 'none' -> (keyboard[4], mouse[2])."""
    s = name.strip().lower().replace(" ", "_").replace("+", "_")
    for a, b in WORD_ALIASES.items():
        s = s.replace(a, b)
    s = "_".join(TOKEN_ALIASES.get(p, p) for p in s.split("_") if p)
    kb = torch.zeros(4)
    ms = torch.zeros(2)
    for cam, val in CAMERA.items():
        if cam in s:
            ms = torch.tensor(val, dtype=torch.float32)
            s = s.replace(cam, "")
            break
    for key, idx in KEYS.items():
        if key in s:
            kb[idx] = 1
    return kb, ms


def fit_frame(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 of any size -> 352x640 by center crop to 11:20 then resize (the official _resizecrop)."""
    h, w = img.shape[:2]
    if h / w > H / W:
        nh = int(w * H / W)
        top = (h - nh) // 2
        img = img[top:top + nh]
    else:
        nw = int(h * W / H)
        left = (w - nw) // 2
        img = img[:, left:left + nw]
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(H, W), mode="bicubic", align_corners=False, antialias=True)
    return t[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()


def _load_weights(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


class Models:
    """The three weights on one device. Built once per (paths, device) by get_models()."""

    def __init__(self, transformer_path: str, vae_path: str, clip_path: str, device, dtype=torch.bfloat16,
                 compile_decoder: bool = False, log=print, config: dict | None = None):
        self.device = torch.device(device)
        self.dtype = dtype
        t0 = time.time()
        if config is None:
            config = json.load(open(os.path.join(HERE, "mg2core", "configs", "universal.json"), encoding="utf-8"))
        cfg = {k: v for k, v in config.items() if not k.startswith("_")}
        model = CausalWanModel.from_config(cfg)  # drops keys the constructor does not take, like the official code
        if transformer_path:
            sd = _load_weights(transformer_path)
            if all(k.startswith("model.") for k in sd):
                sd = {k[len("model."):]: v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Matrix-Game transformer keys mismatch: missing {missing[:5]}... unexpected {unexpected[:5]}...")
            del sd
        self.model = model.eval().requires_grad_(False).to(device=self.device, dtype=dtype)
        self.model.num_frame_per_block = BLOCK
        self.local_attn_size = self.model.local_attn_size
        assert self.local_attn_size != -1, "the streaming path needs a local attention window"
        self.num_layers = len(self.model.blocks)
        log(f"[MatrixGame2] transformer ready in {time.time() - t0:.0f}s")

        sched = FlowMatchScheduler(shift=SHIFT, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(1000, training=True)
        self.scheduler = sched
        ts = torch.cat((sched.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        self.denoising_steps = ts[1000 - torch.tensor(DENOISE, dtype=torch.long)]  # warp_denoising_step: true

        t0 = time.time()
        self.vae = WanVAE(pretrained_path=vae_path or None).to(self.device, dtype)
        dec = VAEDecoderWrapper()
        if vae_path:
            vsd = torch.load(vae_path, map_location="cpu")
            dec.load_state_dict({k: v for k, v in vsd.items() if "decoder." in k or "conv2" in k})
            del vsd
        self.decoder = dec.to(self.device, torch.float16 if self.device.type == "cuda" else dtype).eval().requires_grad_(False)
        if compile_decoder:
            self.decoder.compile(mode="max-autotune-no-cudagraphs")
        log(f"[MatrixGame2] VAE ready in {time.time() - t0:.0f}s")

        t0 = time.time()
        clip, transforms = clip_xlm_roberta_vit_h_14(pretrained=False, return_transforms=True, return_tokenizer=False)
        if clip_path:
            clip.load_state_dict(torch.load(clip_path, map_location="cpu"))
        if hasattr(clip, "textual"):
            clip.textual = None  # only the vision tower is used
        self.clip = clip.eval().requires_grad_(False).to(device=self.device, dtype=dtype)
        self.clip_norm = transforms.transforms[-1]
        log(f"[MatrixGame2] CLIP ready in {time.time() - t0:.0f}s")

    # -- the three model calls -------------------------------------------------------------
    @torch.no_grad()
    def encode_image_context(self, img: torch.Tensor) -> torch.Tensor:
        """img [1,3,1,H,W] in [-1,1] -> CLIP visual tokens [1,257,1280] (encode_video of the official wrapper)."""
        x = img[:, :, 0].float()
        x = F.interpolate(x, size=(self.clip.image_size,) * 2, mode="bicubic", align_corners=False)
        x = self.clip_norm(x.mul(0.5).add(0.5))
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                return self.clip.visual(x.to(self.dtype), use_31_block=True)
        return self.clip.visual(x.to(self.dtype), use_31_block=True)

    @torch.no_grad()
    def predict_x0(self, noisy, cond: dict, timestep, caches: dict, current_start: int) -> torch.Tensor:
        """WanDiffusionWrapper.forward: flow prediction -> x0 (noisy [1,16,F,44,80], timestep [1,F])."""
        flow = self.model(noisy.to(self.model.dtype), t=timestep, **cond, kv_cache=caches["kv"],
                          kv_cache_mouse=caches["mouse"], kv_cache_keyboard=caches["keyboard"],
                          crossattn_cache=caches["cross"], current_start=current_start, cache_start=current_start)
        b, c, f, h, w = flow.shape
        flow2 = flow.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).double()
        xt = noisy.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).double()
        sig = self.scheduler.sigmas.double().to(flow.device)
        ts = self.scheduler.timesteps.double().to(flow.device)
        tid = torch.argmin((ts.unsqueeze(0) - timestep.flatten().double().unsqueeze(1)).abs(), dim=1)
        x0 = xt - sig[tid].reshape(-1, 1, 1, 1) * flow2
        return x0.to(flow.dtype).reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


class SessionFull(RuntimeError):
    pass


class Session:
    """One interactive world: conditioning, KV caches and decoder cache, advanced block by block."""

    def __init__(self, m: Models, frame: np.ndarray, seed: int, sid: str | None = None,
                 encode_latent_frames: int = 25, max_latent_frames: int = 120):
        assert frame.shape == (H, W, 3), f"frame must be {H}x{W}x3, got {frame.shape}"
        self.m, self.id = m, sid or uuid.uuid4().hex[:12]
        dev, dt = m.device, m.dtype
        self.seed = seed
        self.rng = torch.Generator(device=dev).manual_seed(seed)
        img = torch.from_numpy(frame).permute(2, 0, 1).float().div(127.5).sub(1)[None, :, None].to(dev, dt)
        # the official code VAE-encodes [image, zeros x 4*(N-1)]; after the causal receptive field the
        # zero frames encode to the same latent, so encode a prefix and repeat its last frame
        pad = torch.zeros_like(img).repeat(1, 1, 4 * (encode_latent_frames - 1), 1, 1)
        lat = m.vae.encode(torch.cat([img, pad], 2), device=dev, tiled=True, tile_size=(44, 80),
                           tile_stride=(23, 38)).to(dev, dt)
        if max_latent_frames > lat.shape[2]:
            lat = torch.cat([lat, lat[:, :, -1:].expand(-1, -1, max_latent_frames - lat.shape[2], -1, -1)], 2)
        mask = torch.ones_like(lat)
        mask[:, :, 1:] = 0
        self.cond_concat = torch.cat([mask[:, :4], lat], 1)             # [1,20,N,44,80]
        self.visual_context = m.encode_image_context(img).to(dt)        # [1,257,1280]
        self.N = self.cond_concat.shape[2]
        npx = 1 + 4 * (self.N - 1)
        self.keyboard = torch.zeros(1, npx, 4, device=dev, dtype=dt)
        self.mouse = torch.zeros(1, npx, 2, device=dev, dtype=dt)
        self.cursor = 0                                                 # latent frames done
        self.caches = self._init_caches()
        self.vae_cache = [None] * len(ZERO_VAE_CACHE)
        self.frames_done = 0
        self.history: list[str] = []
        self.last_used = time.time()
        self.last_frame = frame

    def _init_caches(self) -> dict:
        m, dev, dt = self.m, self.m.device, self.m.dtype
        n = m.local_attn_size
        kv, mouse, keyboard, cross = [], [], [], []
        for _ in range(m.num_layers):
            kv.append({"k": torch.zeros([1, n * FRAME_SEQ, 12, 128], dtype=dt, device=dev),
                       "v": torch.zeros([1, n * FRAME_SEQ, 12, 128], dtype=dt, device=dev),
                       "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
                       "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)})
            keyboard.append({"k": torch.zeros([1, n, 16, 64], dtype=dt, device=dev),
                             "v": torch.zeros([1, n, 16, 64], dtype=dt, device=dev),
                             "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
                             "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)})
            mouse.append({"k": torch.zeros([FRAME_SEQ, n, 16, 64], dtype=dt, device=dev),
                          "v": torch.zeros([FRAME_SEQ, n, 16, 64], dtype=dt, device=dev),
                          "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
                          "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)})
            cross.append({"k": torch.zeros([1, 257, 12, 128], dtype=dt, device=dev),
                          "v": torch.zeros([1, 257, 12, 128], dtype=dt, device=dev), "is_init": False})
        return dict(kv=kv, mouse=mouse, keyboard=keyboard, cross=cross)

    @property
    def remaining_blocks(self) -> int:
        return (self.N - self.cursor) // BLOCK

    def _cond_for_block(self, kb: torch.Tensor, ms: torch.Tensor) -> dict:
        """cond_current(..., replace=action) of the official pipeline for the block at self.cursor."""
        s = self.cursor
        last = 1 + 4 * (BLOCK - 1) if s == 0 else 4 * BLOCK
        final = 1 + 4 * (s + BLOCK - 1)
        self.mouse[:, final - last:final] = ms.to(self.mouse)[None, None, :].repeat(1, last, 1)
        self.keyboard[:, final - last:final] = kb.to(self.keyboard)[None, None, :].repeat(1, last, 1)
        return dict(cond_concat=self.cond_concat[:, :, s:s + BLOCK], visual_context=self.visual_context,
                    mouse_cond=self.mouse[:, :final], keyboard_cond=self.keyboard[:, :final])

    @torch.no_grad()
    def step(self, actions: list[str]) -> np.ndarray:
        """Advance one block per action; returns the new pixel frames as uint8 [T, 352, 640, 3]."""
        m, dev, dt = self.m, self.m.device, self.m.dtype
        if len(actions) > self.remaining_blocks:
            raise SessionFull(f"world {self.id} has room for {self.remaining_blocks} more blocks")
        out = []
        for a in actions:
            kb, ms = parse_action(a)
            cond = self._cond_for_block(kb, ms)
            start = self.cursor * FRAME_SEQ
            x = torch.randn([1, 16, BLOCK, LH, LW], generator=self.rng, device=dev, dtype=dt)
            x0 = None
            steps = m.denoising_steps
            for i, t in enumerate(steps):
                timestep = torch.full([1, BLOCK], float(t), device=dev, dtype=torch.float32)
                x0 = m.predict_x0(x, cond, timestep, self.caches, start)
                if i < len(steps) - 1:
                    flat = x0.permute(0, 2, 1, 3, 4).reshape(BLOCK, 16, LH, LW)
                    noise = torch.randn(flat.shape, generator=self.rng, device=dev, dtype=dt)
                    nt = torch.full([BLOCK], float(steps[i + 1]), device=dev, dtype=torch.float32)
                    x = m.scheduler.add_noise(flat, noise, nt).reshape(1, BLOCK, 16, LH, LW).permute(0, 2, 1, 3, 4)
            # clean context pass writes this block into the caches
            ctx_t = torch.full([1, BLOCK], float(CONTEXT_NOISE), device=dev, dtype=torch.float32)
            m.predict_x0(x0, cond, ctx_t, self.caches, start)
            dec_dtype = next(m.decoder.parameters()).dtype
            video, self.vae_cache = m.decoder(x0.transpose(1, 2).to(dec_dtype), *self.vae_cache)   # [1,T,3,H,W]
            frames = ((video[0].permute(0, 2, 3, 1).float() + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            out.append(frames)
            self.cursor += BLOCK
            self.frames_done += frames.shape[0]
            self.history.append(a)
        self.last_used = time.time()
        res = np.concatenate(out, 0)
        self.last_frame = res[-1]
        return res

    def info(self) -> dict:
        return dict(session=self.id, seed=self.seed, latent_cursor=self.cursor, latent_capacity=self.N,
                    blocks_left=self.remaining_blocks, frames_done=self.frames_done, fps=FPS,
                    history=self.history, width=W, height=H)


# ----------------------------------------------------------------------------- registries
_MODELS: dict[tuple, Models] = {}
_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()
MAX_SESSIONS = int(os.environ.get("MG2_MAX_SESSIONS", "3"))


def get_models(transformer_path, vae_path, clip_path, device, compile_decoder=False, log=print) -> Models:
    key = (transformer_path, vae_path, clip_path, str(device), compile_decoder)
    with _LOCK:
        if key not in _MODELS:
            _MODELS.clear()
            _SESSIONS.clear()
            _MODELS[key] = Models(transformer_path, vae_path, clip_path, device, compile_decoder=compile_decoder, log=log)
        return _MODELS[key]


def get_session(sid: str) -> Session | None:
    with _LOCK:
        return _SESSIONS.get(sid)


def new_session(m: Models, frame: np.ndarray, seed: int, sid: str | None = None, **kw) -> Session:
    s = Session(m, frame, seed, sid=sid, **kw)
    with _LOCK:
        if len(_SESSIONS) >= MAX_SESSIONS:
            oldest = min(_SESSIONS.values(), key=lambda x: x.last_used)
            _SESSIONS.pop(oldest.id, None)
        _SESSIONS[s.id] = s
    return s


def drop_session(sid: str):
    with _LOCK:
        _SESSIONS.pop(sid, None)
