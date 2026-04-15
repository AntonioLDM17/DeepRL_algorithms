#!/usr/bin/env python3
"""
Generate portfolio media assets for README from local videos.

Outputs (by default under docs/media/):
- banner.png
- architecture_overview.png
- dqn_run.gif
- rainbow_run.gif
- ppo_run.gif
- sac_run.gif
- dqn_thumb.png
- rainbow_thumb.png
- ppo_thumb.png
- sac_thumb.png
- rainbow_ablations.png

Usage:
  python scripts/generate_media.py
  python scripts/generate_media.py --output docs/media --frames 56 --width 320
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw


VIDEO_MAP = {
    "dqn": "videos/walker_train_dqn_new_walker-episode-70000.mp4",
    "rainbow": "videos/walker_train_rainbow-episode-35000.mp4",
    "ppo": "videos/ppo_original_experiment_result.mp4",
    "sac": "videos/sac_original_experiment_result.mp4",
}

RAINBOW_ABLATIONS = [
    ("Full Rainbow", "videos/walker_train_rainbow-episode-35000.mp4"),
    ("No PER", "videos/walker_train_rainbow_noPER-episode-44000.mp4"),
    ("No DIST", "videos/walker_train_rainbow_noDIST-episode-162000.mp4"),
    ("No PER + No DIST", "videos/walker_train_rainbow_noPER_noDIST-episode-166000.mp4"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate docs/media assets from local videos.")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("docs/media"))
    p.add_argument("--frames", type=int, default=56, help="Frames sampled per GIF.")
    p.add_argument("--width", type=int, default=320, help="Target width per sampled frame.")
    p.add_argument("--fps", type=float, default=14.0, help="GIF playback FPS.")
    return p.parse_args()


def _sample_indices(n_frames: int, n_samples: int) -> np.ndarray:
    return np.linspace(0, n_frames - 1, num=min(n_samples, n_frames), dtype=int)


def sample_frames(video_path: Path, max_frames: int, target_w: int) -> list[np.ndarray]:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    reader = iio.get_reader(str(video_path))
    try:
        n = reader.count_frames()
    except Exception:
        n = 0

    if n <= 0:
        frames = [frame for frame in reader]
        n = len(frames)
        idxs = _sample_indices(n, max_frames)
        sampled = [frames[int(i)] for i in idxs]
    else:
        idxs = _sample_indices(n, max_frames)
        sampled = [reader.get_data(int(i)) for i in idxs]

    reader.close()

    out = []
    for fr in sampled:
        im = Image.fromarray(fr)
        h, w = fr.shape[:2]
        resized = im.resize((target_w, int(h * target_w / w)), Image.Resampling.LANCZOS)
        out.append(np.array(resized))
    return out


def save_gif(frames: Iterable[np.ndarray], out_path: Path, fps: float) -> None:
    duration = 1.0 / fps
    iio.mimsave(str(out_path), list(frames), duration=duration, loop=0)


def build_banner(thumbs: dict[str, np.ndarray], out_path: Path) -> None:
    order = ["dqn", "rainbow", "ppo", "sac"]
    blocks = []
    for key in order:
        im = Image.fromarray(thumbs[key]).resize((620, 300), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 262, 160, 300), fill=(0, 0, 0))
        d.text((12, 274), key.upper(), fill=(255, 255, 255))
        blocks.append(np.array(im))

    canvas = np.vstack([np.hstack(blocks[:2]), np.hstack(blocks[2:])])
    im = Image.fromarray(canvas).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle((0, 0, im.width, 96), fill=(11, 14, 21, 210))
    d.text((24, 24), "DeepRL Algorithms — From Pixels to Locomotion", fill=(245, 245, 245))
    d.text((24, 58), "DQN • Rainbow • PPO • SAC on visual MuJoCo control", fill=(174, 198, 235))

    Image.alpha_composite(im, overlay).convert("RGB").save(out_path)


def build_architecture(out_path: Path) -> None:
    W, H = 1360, 700
    im = Image.new("RGB", (W, H), (18, 22, 31))
    d = ImageDraw.Draw(im)

    def box(x0: int, y0: int, x1: int, y1: int, title: str, lines: list[str], color: tuple[int, int, int]) -> None:
        d.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=color, outline=(93, 118, 154), width=2)
        txt = title + "\n" + "\n".join(lines)
        d.multiline_text((x0 + 16, y0 + 14), txt, fill=(236, 241, 249), spacing=4)

    box(30, 70, 320, 210, "MuJoCo Tasks", ["Walker2d / Humanoid", "RGB rendering"], (37, 52, 74))
    box(370, 70, 700, 250, "Shared Visual Pipeline", ["Crop + resize (84x84)", "Frame stack x4", "uint8 storage"], (37, 52, 74))
    box(760, 40, 1330, 290, "Value-based branch", ["Discrete wrappers", "DQN", "Rainbow (+PER/+Noisy/+C51/+n-step)"], (51, 66, 93))
    box(760, 340, 1330, 620, "Actor-critic branch", ["Continuous actions", "PPO", "SAC (entropy regularization)"], (48, 84, 77))
    box(370, 350, 700, 560, "Experiment ops", ["TensorBoard metrics", "Checkpoints", "Evaluation videos"], (37, 52, 74))

    d.line((320, 140, 370, 140), fill=(181, 203, 236), width=4)
    d.line((700, 140, 760, 140), fill=(181, 203, 236), width=4)
    d.line((700, 180, 740, 180, 740, 470, 760, 470), fill=(181, 203, 236), width=4)
    d.line((320, 140, 340, 140, 340, 455, 370, 455), fill=(181, 203, 236), width=4)

    im.save(out_path)


def build_rainbow_ablations(repo_root: Path, out_path: Path, width: int) -> None:
    cells: list[np.ndarray] = []
    for label, rel in RAINBOW_ABLATIONS:
        frame = sample_frames(repo_root / rel, max_frames=1, target_w=width)[0]
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, im.width, 26), fill=(0, 0, 0))
        d.text((8, 6), label, fill=(240, 240, 240))
        cells.append(np.array(im))

    h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        if c.shape[0] < h:
            pad = np.zeros((h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)

    row1 = np.hstack(padded[:2])
    row2 = np.hstack(padded[2:])
    Image.fromarray(np.vstack([row1, row2])).save(out_path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbs: dict[str, np.ndarray] = {}

    for algo, rel in VIDEO_MAP.items():
        frames = sample_frames(repo_root / rel, max_frames=args.frames, target_w=args.width)
        save_gif(frames, output_dir / f"{algo}_run.gif", fps=args.fps)

        thumb = Image.fromarray(frames[len(frames) // 2])
        thumb.save(output_dir / f"{algo}_thumb.png")
        thumbs[algo] = np.array(thumb)

    build_rainbow_ablations(repo_root, output_dir / "rainbow_ablations.png", width=300)
    build_banner(thumbs, output_dir / "banner.png")
    build_architecture(output_dir / "architecture_overview.png")

    print("Generated media assets in:", output_dir)
    for p in sorted(output_dir.glob("*")):
        print(" -", p.relative_to(repo_root))


if __name__ == "__main__":
    main()
