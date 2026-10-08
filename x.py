#!/usr/bin/env python3
"""Multi-model feature extraction for SPARC with strict row alignment.

This script is extraction-only: no training, no SAE/SPARC fitting.
It supports two phases:
- globals_only: extract compact per-image global vectors
- patch_tokens: extract dense patch/token grids into memmap arrays

Design goals:
- Canonical manifest built once and reused by all models/phases
- Deterministic row indices across all outputs
- Sequential model execution (one model in memory at a time)
- Resumable writes via per-model row checkpoints
- Mandatory alignment verification gate before dense phase
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torchvision.transforms as T

import timm
from timm.models import load_checkpoint

import open_clip

from transformers import AutoImageProcessor, AutoModel


# Offline guards: never pull remote content in extraction runs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_size(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0 or unit == "TB":
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{num_bytes} B"


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_flag.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but CUDA is not available")
    return torch.device(device_flag)


@dataclass
class ManifestRow:
    row_idx: int
    rel_path: str
    orig_w: int
    orig_h: int
    orig_mode: str
    status: str


def gather_image_candidates(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    paths = [p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    paths.sort(key=lambda p: str(p.relative_to(corpus_dir)).lower())
    return paths


def prevalidate_manifest(corpus_dir: Path, output_dir: Path, force: bool) -> tuple[Path, Path, int, int]:
    manifest_path = output_dir / "manifest.csv"
    hash_path = output_dir / "manifest.hash"

    if manifest_path.exists() and hash_path.exists() and not force:
        rows = sum(1 for _ in manifest_path.open("r", encoding="utf-8")) - 1
        skipped = 0
        return manifest_path, hash_path, rows, skipped

    candidates = gather_image_candidates(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows: list[ManifestRow] = []
    skipped = 0

    bar = tqdm(candidates, desc="Pre-validating corpus", unit="img")
    for path in bar:
        rel = str(path.relative_to(corpus_dir)).replace("\\", "/")
        try:
            with Image.open(path) as img:
                mode = img.mode
                w, h = img.size
                _ = img.convert("RGB")
            valid_rows.append(
                ManifestRow(
                    row_idx=len(valid_rows),
                    rel_path=rel,
                    orig_w=w,
                    orig_h=h,
                    orig_mode=mode,
                    status="ok",
                )
            )
        except (UnidentifiedImageError, OSError, ValueError):
            skipped += 1
            continue

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_idx", "filename", "orig_w", "orig_h", "mode", "status"])
        for row in valid_rows:
            writer.writerow([row.row_idx, row.rel_path, row.orig_w, row.orig_h, row.orig_mode, row.status])

    manifest_bytes = manifest_path.read_bytes()
    digest = sha256_bytes(manifest_bytes)
    hash_path.write_text(digest + "\n", encoding="utf-8")

    return manifest_path, hash_path, len(valid_rows), skipped


def load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            rows.append(
                ManifestRow(
                    row_idx=int(item["row_idx"]),
                    rel_path=item["filename"],
                    orig_w=int(item["orig_w"]),
                    orig_h=int(item["orig_h"]),
                    orig_mode=item["mode"],
                    status=item["status"],
                )
            )
    return rows


def subset_to_indices(subset_arg: str | None, manifest_rows: Sequence[ManifestRow]) -> list[int] | None:
    if subset_arg is None:
        return None

    path = Path(subset_arg)
    if not path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_arg}")

    content = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not content:
        return []

    is_ints = True
    for line in content:
        if not line.isdigit():
            is_ints = False
            break

    if is_ints:
        idx = sorted({int(v) for v in content})
        max_idx = len(manifest_rows) - 1
        for i in idx:
            if i < 0 or i > max_idx:
                raise ValueError(f"Subset index out of range: {i} not in [0, {max_idx}]")
        return idx

    row_by_name = {row.rel_path: row.row_idx for row in manifest_rows}
    out: set[int] = set()
    missing: list[str] = []
    for name in content:
        if name in row_by_name:
            out.add(row_by_name[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Subset contains filenames not in manifest (first 5): {missing[:5]}")
    return sorted(out)


def image_to_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float().cpu()
    finite_mask = torch.isfinite(x)
    nan_count = int(torch.isnan(x).sum().item())
    inf_count = int(torch.isinf(x).sum().item())
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_ratio": float(finite_mask.float().mean().item()),
    }
    return out


def assert_no_nan_inf(x: torch.Tensor, model_name: str, kind: str) -> None:
    if not torch.isfinite(x).all():
        nan_count = int(torch.isnan(x).sum().item())
        inf_count = int(torch.isinf(x).sum().item())
        raise RuntimeError(
            f"{model_name} produced non-finite {kind}: nan_count={nan_count} inf_count={inf_count}"
        )


class MemmapWriter:
    def __init__(self, path: Path, shape: tuple[int, ...], dtype: np.dtype):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mm = np.memmap(self.path, mode="w+", dtype=self.dtype, shape=self.shape)

    def write_rows(self, row_indices: Sequence[int], batch: np.ndarray) -> None:
        self.mm[np.array(row_indices, dtype=np.int64)] = batch

    def flush(self) -> None:
        self.mm.flush()

    def close(self) -> None:
        self.flush()
        del self.mm


@dataclass
class ModelRunOutput:
    globals_primary: torch.Tensor
    globals_secondary: torch.Tensor | None
    dense_tokens: torch.Tensor | None


class BaseAdapter:
    name: str
    out_dir_name: str

    def __init__(self, checkpoints_root: Path, device: torch.device, batch_size: int):
        self.checkpoints_root = checkpoints_root
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        if self.model is not None:
            del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def run_batch(self, batch_rgb: list[Image.Image], need_dense: bool) -> ModelRunOutput:
        raise NotImplementedError

    def preprocessing_metadata(self) -> dict:
        raise NotImplementedError

    def grid_shape_metadata(self) -> dict:
        raise NotImplementedError

    def global_filename_primary(self) -> str:
        raise NotImplementedError

    def global_filename_secondary(self) -> str | None:
        return None


class DinoV3Adapter(BaseAdapter):
    name = "dinov3"
    out_dir_name = "dinov3"

    def load(self) -> None:
        ckpt = self.checkpoints_root / "dinov3_vitl16"
        self.processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
        self.model = AutoModel.from_pretrained(ckpt, local_files_only=True).to(self.device).eval()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        out = self.processor(images=images, return_tensors="pt")
        return out["pixel_values"].to(self.device)

    def run_batch(self, batch_rgb: list[Image.Image], need_dense: bool) -> ModelRunOutput:
        x = self.preprocess(batch_rgb)
        with torch.no_grad():
            out = self.model(pixel_values=x)
        seq = out.last_hidden_state  # (B, 201, 1024): cls + 4 reg + 196 patch
        cls = seq[:, 0, :]
        patch = seq[:, 1 + self.model.config.num_register_tokens :, :] if need_dense else None
        return ModelRunOutput(globals_primary=cls, globals_secondary=None, dense_tokens=patch)

    def preprocessing_metadata(self) -> dict:
        cfg = self.processor.to_dict()
        return {
            "resolution": 224,
            "source": "AutoImageProcessor local",
            "config": cfg,
        }

    def grid_shape_metadata(self) -> dict:
        return {"h": 14, "w": 14, "n_tokens": 196, "dim": 1024, "has_cls": True, "n_reg": 4}

    def global_filename_primary(self) -> str:
        return "global.pt"


class VitAdapter(BaseAdapter):
    name = "vit"
    out_dir_name = "vit"

    def load(self) -> None:
        ckpt_dir = self.checkpoints_root / "vit_large_supervised"
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False).to(self.device).eval()
        weight = ckpt_dir / "model.safetensors"
        load_checkpoint(self.model, str(weight))

        cfg = self.model.pretrained_cfg
        self._mean = cfg.get("mean", (0.5, 0.5, 0.5))
        self._std = cfg.get("std", (0.5, 0.5, 0.5))
        self._interpolation = cfg.get("interpolation", "bicubic")
        self._crop_pct = float(cfg.get("crop_pct", 0.9))
        self._input_size = cfg.get("input_size", (3, 224, 224))

        resize_size = int(round(self._input_size[1] / self._crop_pct))
        self.transform = T.Compose(
            [
                T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self._input_size[1]),
                T.ToTensor(),
                T.Normalize(mean=self._mean, std=self._std),
            ]
        )

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transform(im) for im in images], dim=0)
        return x.to(self.device)

    def run_batch(self, batch_rgb: list[Image.Image], need_dense: bool) -> ModelRunOutput:
        x = self.preprocess(batch_rgb)
        with torch.no_grad():
            seq = self.model.forward_features(x)  # (B, 197, 1024)
        cls = seq[:, 0, :]
        patch = seq[:, 1:, :] if need_dense else None
        return ModelRunOutput(globals_primary=cls, globals_secondary=None, dense_tokens=patch)

    def preprocessing_metadata(self) -> dict:
        return {
            "resolution": int(self._input_size[1]),
            "resize": int(round(self._input_size[1] / self._crop_pct)),
            "crop_pct": self._crop_pct,
            "interpolation": self._interpolation,
            "mean": list(self._mean),
            "std": list(self._std),
        }

    def grid_shape_metadata(self) -> dict:
        return {"h": 14, "w": 14, "n_tokens": 196, "dim": 1024, "has_cls": True}

    def global_filename_primary(self) -> str:
        return "global.pt"


class ClipAdapter(BaseAdapter):
    name = "clip_img"
    out_dir_name = "clip_img"

    def load(self) -> None:
        ckpt_file = self.checkpoints_root / "clip_vitl14" / "clip_ViT_L_14_datacomp_xl_s13b_b90k.pt"
        self.model = open_clip.create_model("ViT-L-14", pretrained=None)
        state = torch.load(ckpt_file, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(self.device).eval()

        _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
        self.transform = preprocess

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transform(im) for im in images], dim=0)
        return x.to(self.device)

    def _preproj_cls_and_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-projection CLS token (1024) and patch tokens (B,256,1024)."""
        vis = self.model.visual
        y = vis._embeds(x)
        y = vis.transformer(y)
        y = vis.ln_post(y)
        cls = y[:, 0, :]
        patch = y[:, 1:, :]
        return cls, patch

    def run_batch(self, batch_rgb: list[Image.Image], need_dense: bool) -> ModelRunOutput:
        x = self.preprocess(batch_rgb)
        with torch.no_grad():
            post = self.model.encode_image(x, normalize=False)  # (B,768) post-proj
            pre_cls, patch = self._preproj_cls_and_patch(x)
        return ModelRunOutput(globals_primary=post, globals_secondary=pre_cls, dense_tokens=patch if need_dense else None)

    def preprocessing_metadata(self) -> dict:
        return {
            "resolution": 224,
            "source": "open_clip ViT-L-14 preprocess",
            "tag": "datacomp_xl_s13b_b90k",
        }

    def grid_shape_metadata(self) -> dict:
        return {"h": 16, "w": 16, "n_tokens": 256, "dim": 1024, "has_cls": True, "patch_size": 14}

    def global_filename_primary(self) -> str:
        return "global.pt"

    def global_filename_secondary(self) -> str | None:
        return "global_preproj.pt"


class VJepaAdapter(BaseAdapter):
    name = "vjepa21"
    out_dir_name = "vjepa21"

    def load(self) -> None:
        ckpt = self.checkpoints_root / "vjepa21_vitl_384"
        self.model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True).to(self.device).eval()

        self.transform = T.Compose(
            [
                T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        # True still-image mode: T=1, no fake temporal duplication.
        x2d = torch.stack([self.transform(im) for im in images], dim=0)  # (B, C, H, W)
        return x2d.unsqueeze(2).to(self.device)  # (B, C, 1, H, W)

    def run_batch(self, batch_rgb: list[Image.Image], need_dense: bool) -> ModelRunOutput:
        x = self.preprocess(batch_rgb)
        with torch.no_grad():
            out = self.model(x, skip_predictor=True)
        seq = out.last_hidden_state  # expected (B, 576, 1024) in image mode
        mean_pool = seq.mean(dim=1)
        max_pool = seq.max(dim=1).values
        return ModelRunOutput(globals_primary=mean_pool, globals_secondary=max_pool, dense_tokens=seq if need_dense else None)

    def preprocessing_metadata(self) -> dict:
        return {
            "resolution": 384,
            "resize": 384,
            "crop": 384,
            "interpolation": "bicubic",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "note": "Still image mode via tensor shape (B,C,1,384,384), no frame duplication.",
        }

    def grid_shape_metadata(self) -> dict:
        return {
            "h": 24,
            "w": 24,
            "n_tokens": 576,
            "dim": 1024,
            "has_cls": False,
            "mapping": "image-mode token sequence reshaped to 24x24 from patch16 at 384 with T=1",
        }

    def global_filename_primary(self) -> str:
        return "global_meanpool.pt"

    def global_filename_secondary(self) -> str | None:
        return "global_maxpool.pt"


def build_adapter(name: str, checkpoints_root: Path, device: torch.device, batch_size: int) -> BaseAdapter:
    mapping: dict[str, type[BaseAdapter]] = {
        "dinov3": DinoV3Adapter,
        "vit": VitAdapter,
        "clip_img": ClipAdapter,
        "vjepa21": VJepaAdapter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown model: {name}")
    return mapping[name](checkpoints_root, device, batch_size)


def model_order_from_flag(flag: str) -> list[str]:
    if flag:
        names = [x.strip() for x in flag.split(",") if x.strip()]
        return names
    return ["dinov3", "vit", "clip_img", "vjepa21"]


def read_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_model_manifest_hash(model_dir: Path, manifest_hash: str) -> None:
    (model_dir / "manifest.hash").write_text(manifest_hash + "\n", encoding="utf-8")


def save_globals_tensor(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(arr), path)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def run_dummy_forward(
    adapters: list[BaseAdapter],
    corpus_dir: Path,
    manifest_rows: Sequence[ManifestRow],
    output_dir: Path,
) -> None:
    if not manifest_rows:
        raise RuntimeError("Manifest has no valid rows")

    sample_rel = manifest_rows[0].rel_path
    sample_img = image_to_rgb(corpus_dir / sample_rel)

    lines: list[str] = []
    lines.append(f"[{now_ts()}] Dummy-forward check on sample: {sample_rel}")

    for adapter in adapters:
        adapter.load()
        out = adapter.run_batch([sample_img], need_dense=True)

        g1 = tuple(out.globals_primary.shape)
        g2 = tuple(out.globals_secondary.shape) if out.globals_secondary is not None else None
        d = tuple(out.dense_tokens.shape) if out.dense_tokens is not None else None
        lines.append(f"{adapter.name}: global_primary={g1} global_secondary={g2} dense={d}")

        # Hard assertions for expected dimensions to catch errors before full run.
        if adapter.name in {"dinov3", "vit", "vjepa21"}:
            if out.globals_primary.shape[-1] != 1024:
                raise RuntimeError(f"{adapter.name} global dim mismatch: {out.globals_primary.shape[-1]} != 1024")
        if adapter.name == "clip_img":
            if out.globals_primary.shape[-1] != 768:
                raise RuntimeError(f"clip_img primary global dim mismatch: {out.globals_primary.shape[-1]} != 768")
            if out.globals_secondary is None or out.globals_secondary.shape[-1] != 1024:
                raise RuntimeError("clip_img secondary preproj dim mismatch")

        if out.dense_tokens is None:
            raise RuntimeError(f"{adapter.name} dummy dense tokens missing")

        if adapter.name == "dinov3" and out.dense_tokens.shape[1] != 196:
            raise RuntimeError(f"dinov3 dense token count mismatch: {out.dense_tokens.shape[1]} != 196")
        if adapter.name == "vit" and out.dense_tokens.shape[1] != 196:
            raise RuntimeError(f"vit dense token count mismatch: {out.dense_tokens.shape[1]} != 196")
        if adapter.name == "clip_img" and out.dense_tokens.shape[1] != 256:
            raise RuntimeError(f"clip_img dense token count mismatch: {out.dense_tokens.shape[1]} != 256")
        if adapter.name == "vjepa21":
            n_tok = out.dense_tokens.shape[1]
            side = int(math.sqrt(n_tok))
            if side * side != n_tok or n_tok != 576:
                raise RuntimeError(
                    "V-JEPA token mapping ambiguous in image mode; expected 576 tokens (24x24), got "
                    f"{n_tok}. STOP as requested."
                )

        adapter.unload()

    dummy_log = output_dir / "dummy_forward.log"
    dummy_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def get_resume_index(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        value = int(path.read_text(encoding="utf-8").strip())
        return max(0, value)
    except ValueError:
        return 0


def set_resume_index(path: Path, value: int) -> None:
    path.write_text(str(value) + "\n", encoding="utf-8")


def sample_norms(x: torch.Tensor, n: int = 8) -> list[float]:
    x = x.detach().float().cpu()
    if x.ndim != 2:
        return []
    take = min(n, x.shape[0])
    idx = np.linspace(0, x.shape[0] - 1, num=take, dtype=np.int64) if take > 1 else np.array([0], dtype=np.int64)
    vals = torch.linalg.norm(x[idx], dim=1)
    return [float(v.item()) for v in vals]


def maybe_import_verify_alignment(output_dir: Path):
    sys.path.insert(0, str(output_dir))
    import verify_alignment  # type: ignore

    return verify_alignment


def extract_one_model(
    adapter: BaseAdapter,
    corpus_dir: Path,
    output_dir: Path,
    manifest_rows: Sequence[ManifestRow],
    manifest_hash: str,
    globals_only: bool,
    patch_tokens: bool,
    flush_every_batches: int,
    subset_indices: set[int] | None,
) -> dict:
    model_dir = output_dir / adapter.out_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    write_model_manifest_hash(model_dir, manifest_hash)
    write_json(model_dir / "preprocessing.json", adapter.preprocessing_metadata())

    run_dense = patch_tokens
    run_globals = globals_only or (not patch_tokens)

    total_n = len(manifest_rows)

    if subset_indices is not None:
        active_indices = sorted(subset_indices)
    else:
        active_indices = list(range(total_n))

    # Global arrays always shaped by full canonical N for strict row mapping.
    g_primary_path = model_dir / adapter.global_filename_primary()
    g_secondary_name = adapter.global_filename_secondary()
    g_secondary_path = model_dir / g_secondary_name if g_secondary_name else None

    dense_dat_path = model_dir / "patch_tokens.dat"
    dense_meta_path = model_dir / "patch_tokens.meta.json"

    resume_path = model_dir / "resume_state.json"
    state = {"next_pos": 0}
    if resume_path.exists():
        try:
            state = json.loads(resume_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"next_pos": 0}

    # Determine dimensions from one sample in active subset.
    probe_idx = active_indices[0]
    probe_img = image_to_rgb(corpus_dir / manifest_rows[probe_idx].rel_path)
    probe_out = adapter.run_batch([probe_img], need_dense=run_dense)

    g_dim = int(probe_out.globals_primary.shape[-1])
    g2_dim = int(probe_out.globals_secondary.shape[-1]) if probe_out.globals_secondary is not None else None

    if run_dense:
        if probe_out.dense_tokens is None:
            raise RuntimeError(f"{adapter.name} requested dense extraction but adapter returned no dense tokens")
        n_tokens = int(probe_out.dense_tokens.shape[1])
        tok_dim = int(probe_out.dense_tokens.shape[2])
        grid_meta = adapter.grid_shape_metadata() | {"n_tokens": n_tokens, "dim": tok_dim}
        write_json(model_dir / "grid_shape.json", grid_meta)

    # Prepare output buffers.
    g_primary = np.zeros((total_n, g_dim), dtype=np.float16) if run_globals else None
    g_secondary = np.zeros((total_n, g2_dim), dtype=np.float16) if (run_globals and g2_dim is not None) else None

    dense_writer = None
    if run_dense:
        dense_writer = MemmapWriter(dense_dat_path, shape=(total_n, n_tokens, tok_dim), dtype=np.float16)
        write_json(
            dense_meta_path,
            {
                "shape": [total_n, n_tokens, tok_dim],
                "dtype": "float16",
                "row_index_source": "manifest.csv",
            },
        )

    started = time.time()
    processed = 0
    skipped = 0
    flush_count = 0

    next_pos = int(state.get("next_pos", 0))
    if next_pos >= len(active_indices):
        next_pos = 0

    bar = tqdm(total=len(active_indices), desc=f"{adapter.name} extraction", unit="img")
    bar.update(next_pos)

    B = adapter.batch_size
    for pos in range(next_pos, len(active_indices), B):
        chunk_idx = active_indices[pos : pos + B]
        images: list[Image.Image] = []
        good_idx: list[int] = []

        for ridx in chunk_idx:
            try:
                images.append(image_to_rgb(corpus_dir / manifest_rows[ridx].rel_path))
                good_idx.append(ridx)
            except Exception:
                # Extremely rare because pre-validation already succeeded.
                skipped += 1
                continue

        if not images:
            set_resume_index(model_dir / "resume_pos.txt", pos + len(chunk_idx))
            continue

        out = adapter.run_batch(images, need_dense=run_dense)

        assert_no_nan_inf(out.globals_primary, adapter.name, "global_primary")

        if run_globals and g_primary is not None:
            g_primary[np.array(good_idx, dtype=np.int64)] = out.globals_primary.detach().cpu().to(torch.float16).numpy()

        if run_globals and g_secondary is not None and out.globals_secondary is not None:
            assert_no_nan_inf(out.globals_secondary, adapter.name, "global_secondary")
            g_secondary[np.array(good_idx, dtype=np.int64)] = out.globals_secondary.detach().cpu().to(torch.float16).numpy()

        if run_dense and dense_writer is not None and out.dense_tokens is not None:
            assert_no_nan_inf(out.dense_tokens, adapter.name, "dense_tokens")
            dense_np = out.dense_tokens.detach().cpu().to(torch.float16).numpy()
            dense_writer.write_rows(good_idx, dense_np)

        processed += len(good_idx)
        bar.update(len(chunk_idx))

        # Periodic flush and checkpoint for resumability.
        flush_count += 1
        if flush_count % max(1, flush_every_batches) == 0:
            if dense_writer is not None:
                dense_writer.flush()
            resume_path.write_text(json.dumps({"next_pos": pos + len(chunk_idx)}, indent=2) + "\n", encoding="utf-8")
            print(
                f"[{now_ts()}] flushed {adapter.name} at manifest position {pos + len(chunk_idx)}/{len(active_indices)}"
            )

    bar.close()

    if dense_writer is not None:
        dense_writer.close()

    if run_globals and g_primary is not None:
        save_globals_tensor(g_primary_path, g_primary)
    if run_globals and g_secondary is not None and g_secondary_path is not None:
        save_globals_tensor(g_secondary_path, g_secondary)

    # Cleanup resume state once model finished.
    resume_path.write_text(json.dumps({"next_pos": len(active_indices), "done": True}, indent=2) + "\n", encoding="utf-8")

    elapsed = time.time() - started

    result = {
        "model": adapter.name,
        "processed": processed,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "global_primary": str(g_primary_path) if run_globals else None,
        "global_primary_size": file_size(g_primary_path) if run_globals else 0,
        "global_secondary": str(g_secondary_path) if (run_globals and g_secondary_path is not None) else None,
        "global_secondary_size": file_size(g_secondary_path) if (run_globals and g_secondary_path is not None) else 0,
        "dense_path": str(dense_dat_path) if run_dense else None,
        "dense_size": file_size(dense_dat_path) if run_dense else 0,
        "n_manifest": total_n,
    }

    # Per-model sanity report (globals).
    if run_globals and g_primary is not None:
        g_primary_t = torch.from_numpy(g_primary)
        result["global_primary_stats"] = tensor_stats(g_primary_t)
        result["global_primary_norm_samples"] = sample_norms(g_primary_t, 8)
        if g_secondary is not None:
            g_secondary_t = torch.from_numpy(g_secondary)
            result["global_secondary_stats"] = tensor_stats(g_secondary_t)
            result["global_secondary_norm_samples"] = sample_norms(g_secondary_t, 8)

    return result


def copy_verify_script(output_dir: Path) -> None:
    src = Path(__file__).parent / "verify_alignment.py"
    dst = output_dir / "verify_alignment.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def write_extraction_report(
    output_dir: Path,
    checkpoint_report_path: Path,
    manifest_n: int,
    manifest_skips: int,
    phase: str,
    run_results: list[dict],
    alignment_ok: bool,
) -> None:
    lines: list[str] = []
    lines.append("# EXTRACTION REPORT")
    lines.append("")
    lines.append(f"- Timestamp: {now_ts()}")
    lines.append(f"- Phase: {phase}")
    lines.append(f"- Checkpoint report source: {checkpoint_report_path}")
    lines.append(f"- Final canonical N: {manifest_n}")
    lines.append(f"- Pre-validation skip count: {manifest_skips}")
    lines.append("- Skip policy: only rows that pass pre-validation are included; row indices are fixed by manifest.csv.")
    lines.append(f"- Alignment check: {'PASS' if alignment_ok else 'FAIL'}")
    lines.append("")

    for item in run_results:
        lines.append(f"## {item['model']}")
        lines.append(f"- Processed: {item['processed']}")
        lines.append(f"- Skipped during extraction: {item['skipped']}")
        lines.append(f"- Time: {item['elapsed_sec']:.2f}s")
        if item.get("global_primary"):
            lines.append(f"- Global primary file: {item['global_primary']} ({human_size(item['global_primary_size'])})")
        if item.get("global_secondary"):
            lines.append(f"- Global secondary file: {item['global_secondary']} ({human_size(item['global_secondary_size'])})")
        if item.get("dense_path"):
            lines.append(f"- Dense token file: {item['dense_path']} ({human_size(item['dense_size'])})")
        if item.get("global_primary_stats"):
            st = item["global_primary_stats"]
            lines.append(
                "- Global primary stats: "
                f"shape={st['shape']} dtype={st['dtype']} mean={st['mean']:.6f} std={st['std']:.6f} "
                f"min={st['min']:.6f} max={st['max']:.6f} nan={st['nan_count']} inf={st['inf_count']}"
            )
            lines.append(f"- Sample primary norms: {item.get('global_primary_norm_samples', [])}")
        if item.get("global_secondary_stats"):
            st = item["global_secondary_stats"]
            lines.append(
                "- Global secondary stats: "
                f"shape={st['shape']} dtype={st['dtype']} mean={st['mean']:.6f} std={st['std']:.6f} "
                f"min={st['min']:.6f} max={st['max']:.6f} nan={st['nan_count']} inf={st['inf_count']}"
            )
            lines.append(f"- Sample secondary norms: {item.get('global_secondary_norm_samples', [])}")
        lines.append("")

    (output_dir / "EXTRACTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_alignment_check(output_dir: Path, sample_k: int, seed: int) -> tuple[bool, str]:
    verify_alignment = maybe_import_verify_alignment(output_dir)
    ok, msg = verify_alignment.verify_alignment_main(str(output_dir), sample_k=sample_k, seed=seed)
    return ok, msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust, resumable multi-model extraction for SPARC")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Directory containing corpus images")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Local checkpoints root")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Extraction output root")

    parser.add_argument("--globals_only", action="store_true", help="Phase 1: global vectors only")
    parser.add_argument("--patch_tokens", action="store_true", help="Phase 2: dense tokens")
    parser.add_argument("--prevalidate_only", action="store_true", help="Only pre-validate and write manifest")
    parser.add_argument("--dummy_check_only", action="store_true", help="Only run dummy-forward checks")

    parser.add_argument("--model", type=str, default="", help="Run a single model name or comma-separated order")
    parser.add_argument("--device", type=str, default="auto", help="Device, e.g. auto|cpu|cuda:0|cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0, help="Reserved for future dataloader use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush_every_batches", type=int, default=100)
    parser.add_argument("--subset", type=str, default=None, help="Optional file with row indices or filenames")
    parser.add_argument("--force_rebuild_manifest", action="store_true")
    parser.add_argument("--sample_k", type=int, default=10, help="Random row checks for alignment")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation after dummy check")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    corpus_dir = Path(args.corpus_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_report = checkpoints_dir / "DOWNLOAD_REPORT.md"
    if not checkpoint_report.exists():
        raise FileNotFoundError(f"Checkpoint report missing: {checkpoint_report}")

    device = resolve_device(args.device)

    print(f"[{now_ts()}] Using device: {device}")
    print(f"[{now_ts()}] Corpus dir: {corpus_dir}")
    print(f"[{now_ts()}] Checkpoints dir: {checkpoints_dir}")
    print(f"[{now_ts()}] Output dir: {output_dir}")

    manifest_path, hash_path, manifest_n, skipped = prevalidate_manifest(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        force=args.force_rebuild_manifest,
    )
    manifest_hash = read_hash(hash_path)
    rows = load_manifest_rows(manifest_path)

    print(f"[{now_ts()}] Canonical manifest rows: {manifest_n}, skipped in pre-validation: {skipped}")
    print(f"[{now_ts()}] Manifest hash: {manifest_hash}")

    copy_verify_script(output_dir)

    if args.prevalidate_only:
        print("Pre-validation completed; exiting due to --prevalidate_only")
        return 0

    model_names = model_order_from_flag(args.model)

    if any(name not in {"dinov3", "vit", "clip_img", "vjepa21"} for name in model_names):
        raise ValueError(f"Invalid model selection/order: {model_names}")

    adapters = [build_adapter(name, checkpoints_dir, device, args.batch_size) for name in model_names]

    subset_idx = subset_to_indices(args.subset, rows)
    subset_set = set(subset_idx) if subset_idx is not None else None
    if subset_set is not None:
        print(f"[{now_ts()}] Dense/global extraction restricted to subset of {len(subset_set)} rows")

    # Dummy-forward gate before full corpus work.
    run_dummy_forward(adapters, corpus_dir, rows, output_dir)

    if not args.yes:
        print("Dummy-forward checks complete. Confirm to continue full extraction [y/N]: ", end="", flush=True)
        reply = sys.stdin.readline().strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted by user after dummy check")
            return 2

    if args.dummy_check_only:
        print("Dummy-forward completed; exiting due to --dummy_check_only")
        return 0

    if args.patch_tokens:
        # Mandatory alignment check gate for Phase 2.
        ok, msg = run_alignment_check(output_dir, sample_k=args.sample_k, seed=args.seed)
        print(msg)
        if not ok:
            raise RuntimeError("Alignment gate failed before Phase 2 dense extraction. STOP.")

    run_results: list[dict] = []

    total_models = len(adapters)
    for idx, adapter in enumerate(adapters, start=1):
        phase_name = "dense tokens" if args.patch_tokens else "globals"
        print(f"\n=== Extracting {adapter.name} [{idx}/{total_models}] - {phase_name} ===")

        adapter.load()
        started = time.time()
        out = extract_one_model(
            adapter=adapter,
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            manifest_rows=rows,
            manifest_hash=manifest_hash,
            globals_only=args.globals_only or (not args.patch_tokens),
            patch_tokens=args.patch_tokens,
            flush_every_batches=max(1, args.flush_every_batches),
            subset_indices=subset_set,
        )
        run_results.append(out)

        adapter.unload()
        took = time.time() - started

        print(
            f"Completed {adapter.name}: processed={out['processed']} skipped={out['skipped']} "
            f"time={took:.2f}s"
        )
        print(
            f"Alignment-so-far: model N={out['n_manifest']} canonical N={manifest_n} "
            f"status={'OK' if out['n_manifest'] == manifest_n else 'MISMATCH'}"
        )

        if out["n_manifest"] != manifest_n:
            raise RuntimeError(
                f"Alignment drift detected immediately after {adapter.name}: model N={out['n_manifest']} vs canonical N={manifest_n}"
            )

    # Final alignment verification is mandatory.
    ok, msg = run_alignment_check(output_dir, sample_k=args.sample_k, seed=args.seed)
    print(msg)
    if not ok:
        raise RuntimeError("Final alignment verification failed. Refusing to trust outputs.")

    phase = "patch_tokens" if args.patch_tokens else "globals_only"
    write_extraction_report(
        output_dir=output_dir,
        checkpoint_report_path=checkpoint_report,
        manifest_n=manifest_n,
        manifest_skips=skipped,
        phase=phase,
        run_results=run_results,
        alignment_ok=ok,
    )

    print(f"\n[{now_ts()}] Extraction complete. Report: {output_dir / 'EXTRACTION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        raise
