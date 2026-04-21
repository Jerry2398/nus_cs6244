"""Convert kortex episode folders into HDF5.

Input layout (per episode directory, e.g. ``kortex/episode_0/``)::

    camera_info.json
    rgb/rgb_{t}.jpg              # 720x1280 RGB, t = 0..T-1
    depth/depth_{t}.npy          # 480x848 uint16
    inhand_rgb/...               # optional, same naming
    inhand_depth/...             # optional, same naming
    action_ee.npy                # (T, 7)
    ee_poses.npy                 # (T, 7)
    joint_positions.npy          # (T, 13)
    joint_velocities.npy         # (T, 13)
    joint_accelerations.npy      # (T, 13)
    joint_torques.npy            # (T, 13)
    controller_btn_states.npy    # (T, 6)
    stop_signal.npy              # (T,)

Usage::

    # Combined file with all demos under /data/demo_<i> (+ one metainfo.json)
    python convert_to_hdf5.py --input kortex --output kortex_dataset.hdf5

    # One HDF5 per episode (+ one demo_<i>_metainfo.json per episode)
    python convert_to_hdf5.py --input kortex --output out_dir --per-episode

    # Provide task labels written into metainfo
    python convert_to_hdf5.py --input kortex --output kortex_dataset.hdf5 \
        --task-description "pick up the red block" --task-nouns robot block

A metainfo JSON is written next to every HDF5. Its schema matches
``inspect_export_code.py`` by default (``--metainfo-style parity``) and
falls back to ``null``/``[]`` for fields that do not exist in real-robot
data (e.g. ``initial_state_seed``, ``exo_boxes``). Use
``--metainfo-style real`` to write a real-robot-friendly metainfo that
only includes fields we actually have.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Iterable

import h5py
import numpy as np
from PIL import Image


NPY_FIELDS = {
    "action_ee": "actions",
    "ee_poses": "ee_pose",
    "joint_positions": "joint_positions",
    "joint_velocities": "joint_velocities",
    "joint_accelerations": "joint_accelerations",
    "joint_torques": "joint_torques",
    "controller_btn_states": "controller_btn_states",
    "stop_signal": "stop_signal",
}


def _numeric_key(path: str) -> int:
    m = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse frame index from {path!r}")
    return int(m.group(1))


def _sorted_frames(folder: str, ext: str) -> list[str]:
    files = glob(os.path.join(folder, f"*{ext}"))
    files.sort(key=_numeric_key)
    return files


def _load_rgb_stack(folder: str) -> np.ndarray | None:
    files = _sorted_frames(folder, ".jpg")
    if not files:
        return None
    first = np.array(Image.open(files[0]).convert("RGB"))
    h, w, c = first.shape
    arr = np.empty((len(files), h, w, c), dtype=np.uint8)
    arr[0] = first
    for i, p in enumerate(files[1:], start=1):
        img = np.array(Image.open(p).convert("RGB"))
        if img.shape != (h, w, c):
            raise ValueError(f"RGB size changed at {p}: {img.shape} vs {(h, w, c)}")
        arr[i] = img
    return arr


def _load_depth_stack(folder: str) -> np.ndarray | None:
    files = _sorted_frames(folder, ".npy")
    if not files:
        return None
    first = np.load(files[0])
    arr = np.empty((len(files), *first.shape), dtype=first.dtype)
    arr[0] = first
    for i, p in enumerate(files[1:], start=1):
        d = np.load(p)
        if d.shape != first.shape or d.dtype != first.dtype:
            raise ValueError(
                f"Depth mismatch at {p}: {d.shape}/{d.dtype} vs {first.shape}/{first.dtype}"
            )
        arr[i] = d
    return arr


def _check_length(name: str, arr: np.ndarray, expected: int) -> None:
    if arr.shape[0] != expected:
        raise ValueError(
            f"Length mismatch for {name}: got {arr.shape[0]}, expected {expected}"
        )


def build_metainfo_entry(
    episode_dir: str,
    T: int,
    *,
    style: str,
    task_description: str,
    task_nouns: list[str],
) -> dict:
    """Build the per-episode metainfo dict.

    ``style='parity'`` mirrors ``inspect_export_code.py`` keys exactly,
    using ``null``/``[]`` for fields unavailable in real-robot data.
    ``style='real'`` only includes fields we actually have.
    """

    joint_positions_path = os.path.join(episode_dir, "joint_positions.npy")
    ee_poses_path = os.path.join(episode_dir, "ee_poses.npy")
    initial_joint_positions = np.load(joint_positions_path)[0].tolist()
    initial_ee_pose = np.load(ee_poses_path)[0].tolist()

    if style == "parity":
        return {
            "success": True,
            "initial_state": initial_joint_positions,
            "initial_state_seed": None,
            "task_nouns": list(task_nouns),
            "task_description": task_description,
            "exo_boxes": [],
        }
    if style == "real":
        return {
            "success": True,
            "num_samples": T,
            "episode_name": os.path.basename(os.path.normpath(episode_dir)),
            "source_path": os.path.abspath(episode_dir),
            "initial_joint_positions": initial_joint_positions,
            "initial_ee_pose": initial_ee_pose,
            "task_description": task_description,
            "task_nouns": list(task_nouns),
        }
    raise ValueError(f"Unknown metainfo style: {style!r}")


def _write_metainfo(
    out_path: str,
    entries: list[tuple[str, int, str]],
    *,
    style: str,
    task_description: str,
    task_nouns: list[str],
) -> None:
    """Write a metainfo JSON shaped like ``inspect_export_code.py``::

        { task_key: { episode_key: { ...fields... } } }

    ``entries`` is a list of ``(episode_dir, T, episode_key)`` tuples.
    """

    task_key = task_description.replace(" ", "_") if task_description else "unlabeled"
    payload: dict = {task_key: {}}
    for episode_dir, T, episode_key in entries:
        payload[task_key][episode_key] = build_metainfo_entry(
            episode_dir,
            T,
            style=style,
            task_description=task_description,
            task_nouns=task_nouns,
        )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_camera_info(group: h5py.Group, camera_info_path: str) -> None:
    with open(camera_info_path, "r") as f:
        info = json.load(f)
    cam = group.create_group("camera_info")
    cam.attrs["height"] = int(info["height"])
    cam.attrs["width"] = int(info["width"])
    cam.attrs["distortion_model"] = info.get("distortion_model", "")
    cam.create_dataset("K", data=np.asarray(info["K"], dtype=np.float64).reshape(3, 3))
    cam.create_dataset("D", data=np.asarray(info["D"], dtype=np.float64))
    cam.create_dataset("R", data=np.asarray(info["R"], dtype=np.float64).reshape(3, 3))
    cam.create_dataset("P", data=np.asarray(info["P"], dtype=np.float64).reshape(3, 4))


def _image_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    # Chunk along time with 1 frame to allow random access without loading all.
    return (1, *shape[1:])


def write_episode_group(
    group: h5py.Group,
    episode_dir: str,
    *,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> int:
    """Write one episode into ``group`` (a /data/demo_<i> group). Returns T."""

    npy = {
        key: np.load(os.path.join(episode_dir, f"{key}.npy"))
        for key in NPY_FIELDS
    }

    T = npy["action_ee"].shape[0]
    for key, arr in npy.items():
        _check_length(key, arr, T)

    rgb = _load_rgb_stack(os.path.join(episode_dir, "rgb"))
    depth = _load_depth_stack(os.path.join(episode_dir, "depth"))
    if rgb is None or depth is None:
        raise ValueError(f"Missing rgb/ or depth/ in {episode_dir}")
    _check_length("rgb", rgb, T)
    _check_length("depth", depth, T)

    inhand_rgb = _load_rgb_stack(os.path.join(episode_dir, "inhand_rgb"))
    inhand_depth = _load_depth_stack(os.path.join(episode_dir, "inhand_depth"))
    if inhand_rgb is not None:
        _check_length("inhand_rgb", inhand_rgb, T)
    if inhand_depth is not None:
        _check_length("inhand_depth", inhand_depth, T)

    group.attrs["num_samples"] = T
    group.attrs["episode_name"] = os.path.basename(os.path.normpath(episode_dir))

    obs = group.create_group("obs")

    obs.create_dataset(
        "rgb",
        data=rgb,
        dtype=np.uint8,
        chunks=_image_chunks(rgb.shape),
        compression=compression,
        compression_opts=compression_opts if compression else None,
    )
    obs.create_dataset(
        "depth",
        data=depth,
        dtype=depth.dtype,
        chunks=_image_chunks(depth.shape),
        compression=compression,
        compression_opts=compression_opts if compression else None,
    )
    if inhand_rgb is not None:
        obs.create_dataset(
            "inhand_rgb",
            data=inhand_rgb,
            chunks=_image_chunks(inhand_rgb.shape),
            compression=compression,
            compression_opts=compression_opts if compression else None,
        )
    if inhand_depth is not None:
        obs.create_dataset(
            "inhand_depth",
            data=inhand_depth,
            chunks=_image_chunks(inhand_depth.shape),
            compression=compression,
            compression_opts=compression_opts if compression else None,
        )

    for key, out_name in NPY_FIELDS.items():
        if out_name in ("actions", "stop_signal"):
            continue
        obs.create_dataset(out_name, data=npy[key])

    group.create_dataset("actions", data=npy["action_ee"])
    group.create_dataset("stop_signal", data=npy["stop_signal"])

    dones = np.zeros(T, dtype=np.uint8)
    dones[-1] = 1
    group.create_dataset("dones", data=dones)

    _write_camera_info(group, os.path.join(episode_dir, "camera_info.json"))

    return T


def discover_episode_dirs(root: str) -> list[str]:
    entries = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.startswith("episode_") and os.path.isdir(os.path.join(root, name))
    ]
    entries.sort(key=lambda p: _numeric_key(p + ".0"))
    return entries


def convert_combined(
    episode_dirs: Iterable[str],
    output_path: str,
    *,
    compression: str | None,
    compression_opts: int,
    metainfo_style: str,
    task_description: str,
    task_nouns: list[str],
) -> None:
    episode_dirs = list(episode_dirs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    entries: list[tuple[str, int, str]] = []
    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["num_demos"] = len(episode_dirs)
        for i, ep in enumerate(episode_dirs):
            demo = data.create_group(f"demo_{i}")
            T = write_episode_group(
                demo,
                ep,
                compression=compression,
                compression_opts=compression_opts,
            )
            entries.append((ep, T, f"demo_{i}"))
            print(f"[combined] demo_{i} <- {ep}: T={T}")

    meta_path = _metainfo_path_for(output_path)
    _write_metainfo(
        meta_path,
        entries,
        style=metainfo_style,
        task_description=task_description,
        task_nouns=task_nouns,
    )
    print(f"[combined] metainfo -> {meta_path}")


def convert_per_episode(
    episode_dirs: Iterable[str],
    output_dir: str,
    *,
    compression: str | None,
    compression_opts: int,
    metainfo_style: str,
    task_description: str,
    task_nouns: list[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, ep in enumerate(episode_dirs):
        out = os.path.join(output_dir, f"demo_{i}.hdf5")
        with h5py.File(out, "w") as f:
            data = f.create_group("data")
            demo = data.create_group(f"demo_{i}")
            T = write_episode_group(
                demo,
                ep,
                compression=compression,
                compression_opts=compression_opts,
            )
        meta_path = os.path.join(output_dir, f"demo_{i}_metainfo.json")
        _write_metainfo(
            meta_path,
            [(ep, T, f"demo_{i}")],
            style=metainfo_style,
            task_description=task_description,
            task_nouns=task_nouns,
        )
        print(f"[per-episode] {out}: T={T} (metainfo -> {meta_path})")


def _metainfo_path_for(hdf5_path: str) -> str:
    base, _ = os.path.splitext(hdf5_path)
    return base + "_metainfo.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder containing episode_* subdirs")
    parser.add_argument("--output", required=True, help="Output HDF5 file or (with --per-episode) directory")
    parser.add_argument("--per-episode", action="store_true", help="Write one demo_<i>.hdf5 per episode")
    parser.add_argument("--no-compression", action="store_true", help="Disable gzip compression for images")
    parser.add_argument("--gzip-level", type=int, default=4, help="gzip level 0-9 (default 4)")
    parser.add_argument(
        "--metainfo-style",
        choices=("parity", "real"),
        default="parity",
        help="'parity': match inspect_export_code.py keys; 'real': real-robot-friendly keys",
    )
    parser.add_argument(
        "--task-description",
        default="",
        help="Task description string stored in metainfo",
    )
    parser.add_argument(
        "--task-nouns",
        nargs="*",
        default=[],
        help="Task nouns (space separated) stored in metainfo",
    )
    args = parser.parse_args()

    compression = None if args.no_compression else "gzip"
    eps = discover_episode_dirs(args.input)
    if not eps:
        raise SystemExit(f"No episode_* directories found under {args.input}")

    common_kwargs = dict(
        compression=compression,
        compression_opts=args.gzip_level,
        metainfo_style=args.metainfo_style,
        task_description=args.task_description,
        task_nouns=list(args.task_nouns),
    )
    if args.per_episode:
        convert_per_episode(eps, args.output, **common_kwargs)
    else:
        convert_combined(eps, args.output, **common_kwargs)


if __name__ == "__main__":
    main()
