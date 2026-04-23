"""Read images from the input folder, run the seven DIP methods, save figures to output/.

Name files in "input" as "1_*.ext" through "7_*.ext"; method `n` runs only on files
prefixed with "n_" (1 = negative, 2 = gamma, …, 7 = bit planes).

If the filename ends with "_gray", the image is converted to
grayscale (2D) before any processing—use. (e.g. "7_..._gray.jpg").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/demo.py` from any working directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.methods import (
    bit_planes,
    contrast_stretch,
    gamma_correction,
    histogram_equalize,
    image_negative,
    intensity_level_slicing,
    logarithmic_transform,
    reconstruct_from_planes,
)


def to_uint8_sk(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if x.size else 1.0
    if m <= 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = float(x.min()), float(x.max())
    y = (x - lo) * (255.0 / (hi - lo + 1e-9))
    return y.astype(np.uint8)


def _as_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0 + 1e-6:
            return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return to_uint8_sk(img)
    if img.dtype != np.uint8:
        return img.astype(np.uint8)
    return img


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    if bgr.shape[2] == 4:
        return cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_gray_flag(path: Path) -> bool:
    return path.stem.lower().endswith("_gray")


def _rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB (or RGBA, first 3 ch) to single-channel uint8, ITU-BT.601-like weights in OpenCV."""
    base = rgb[:, :, :3] if rgb.shape[2] == 4 else rgb
    return cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)


def load_from_path(p: Path) -> tuple[str, np.ndarray] | None:
    """Load one image; return (stem, uint8 RGB 3D or 2D gray). Fails if unreadable."""
    bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        return None
    as_gray = _load_gray_flag(p)
    if bgr.ndim == 2:
        return p.stem, _as_uint8(bgr)
    img = _as_uint8(_bgr_to_rgb(bgr))
    if as_gray:
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = _rgb_to_gray_u8(img)
    return p.stem, img


def list_input_for_method(data_dir: Path, method_index: int) -> list[Path]:
    """Paths whose filename starts with ``f'{method_index}_'``."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not data_dir.is_dir():
        return []
    pre = f"{method_index}_"
    out: list[Path] = []
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file() and p.name.startswith(pre):
            out.append(p)
    return out


def count_input_images(data_dir: Path) -> int:
    return sum(
        len(list_input_for_method(data_dir, n)) for n in range(1, 8)
    )


def save_side_by_side(
    path: Path,
    a: np.ndarray,
    b: np.ndarray,
    title_a: str = "input",
    title_b: str = "output",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, im, t in ((axes[0], a, title_a), (axes[1], b, title_b)):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_bit_plane_montage(
    path: Path,
    orig: np.ndarray,
    planes: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_gray = orig.ndim == 2 or (orig.ndim == 3 and orig.shape[2] == 1)

    if is_gray and planes.ndim == 3:
        g = orig.squeeze()
        fig, ax = plt.subplots(3, 3, figsize=(9, 9))
        ax[0, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
        ax[0, 0].set_title("input")
        ax[0, 0].axis("off")
        pos = [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        for k in range(8):
            r, c = pos[k]
            ax[r, c].imshow(planes[..., k], cmap="gray", vmin=0, vmax=255)
            ax[r, c].set_title(f"bit {k}")
            ax[r, c].axis("off")
        fig.suptitle("grayscale: bit planes 0-7", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    if planes.ndim == 4 and planes.shape[2] == 3:
        fig = plt.figure(figsize=(16, 7))
        ax_in = plt.subplot2grid((4, 8), (0, 0), colspan=8)
        ax_in.imshow(orig)
        ax_in.set_title("input RGB")
        ax_in.axis("off")
        labels = "RGB"
        for c in range(3):
            for k in range(8):
                a = plt.subplot2grid((4, 8), (c + 1, k))
                a.imshow(planes[:, :, c, k], cmap="gray", vmin=0, vmax=255)
                a.set_title(f"{labels[c]}{k}", fontsize=8)
                a.axis("off")
        fig.suptitle("RGB: bit planes per channel", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    fig, a = plt.subplots(1, 1, figsize=(4, 4))
    a.text(0.5, 0.5, "unsupported bit plane layout", ha="center", va="center")
    a.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_demo(
    out_dir: Path,
    data_dir: Path,
    *,
    gamma: float = 2.2,
    slice_low: int = 80,
    slice_high: int = 180,
) -> None:
    """For each n in 1..7, run the matching method on every ``n_*.ext`` file in ``data_dir``."""
    if not data_dir.is_dir():
        raise SystemExit(f"input directory not found: {data_dir.resolve()}")
    if count_input_images(data_dir) == 0:
        raise SystemExit(
            f"no image files found in {data_dir.resolve()} with names 1_*.ext … 7_*.ext"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for method_index, paths in [
        (1, list_input_for_method(data_dir, 1)),
        (2, list_input_for_method(data_dir, 2)),
        (3, list_input_for_method(data_dir, 3)),
        (4, list_input_for_method(data_dir, 4)),
        (5, list_input_for_method(data_dir, 5)),
        (6, list_input_for_method(data_dir, 6)),
        (7, list_input_for_method(data_dir, 7)),
    ]:
        if not paths:
            continue
        for p in paths:
            got = load_from_path(p)
            if got is None:
                continue
            name, im = got
            if method_index == 1:
                save_side_by_side(
                    out_dir / f"01_negative_{name}.png",
                    im,
                    image_negative(im),
                    "input",
                    "negative",
                )
            elif method_index == 2:
                save_side_by_side(
                    out_dir / f"02_gamma_{name}.png",
                    im,
                    gamma_correction(im, gamma=gamma),
                    "input",
                    f"gamma: {gamma}",
                )
            elif method_index == 3:
                save_side_by_side(
                    out_dir / f"03_log_{name}.png",
                    im,
                    logarithmic_transform(im),
                    "input",
                    "log",
                )
            elif method_index == 4:
                save_side_by_side(
                    out_dir / f"04_contrast_stretch_{name}.png",
                    im,
                    contrast_stretch(im, per_channel=True),
                    "input",
                    "stretch",
                )
            elif method_index == 5:
                save_side_by_side(
                    out_dir / f"05_hist_eq_{name}.png",
                    im,
                    histogram_equalize(im),
                    "input",
                    "hist eq (Y in YUV)",
                )
            elif method_index == 6:
                save_side_by_side(
                    out_dir / f"06_level_slice_{name}.png",
                    im,
                    intensity_level_slicing(
                        im,
                        slice_low,
                        slice_high,
                        preserve_background=False,
                        use_value_channel=True,
                    ),
                    "input",
                    f"slice V in [{slice_low},{slice_high}]",
                )
            else:
                bp = bit_planes(im)
                save_bit_plane_montage(
                    out_dir / f"07_bitplanes_{name}.png", im, bp
                )
                r = reconstruct_from_planes(bp)
                save_side_by_side(
                    out_dir / f"07_reconstruct_all_bits_{name}.png",
                    im,
                    r,
                    "input",
                    "recon bits 0-7",
                )
    print(f"Wrote results under {out_dir.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run elementary DIP on images in a folder. "
            "Files 1_*.ext … 7_*.ext map to the seven methods; output goes under ./output by default."
        )
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="output directory (default: ./output)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("input"),
        help="folder of input images (default: ./input); use names 1_… through 7_…",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="gamma for method 2 (default: 2.2)",
    )
    ap.add_argument(
        "--slice-low",
        type=int,
        default=80,
        help="intensity slice lower bound for method 6 (default: 80)",
    )
    ap.add_argument(
        "--slice-high",
        type=int,
        default=180,
        help="intensity slice upper bound for method 6 (default: 180)",
    )
    args = ap.parse_args()
    run_demo(
        args.out,
        args.data_dir,
        gamma=args.gamma,
        slice_low=args.slice_low,
        slice_high=args.slice_high,
    )


if __name__ == "__main__":
    main()
