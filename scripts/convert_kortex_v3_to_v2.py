#!/usr/bin/env python
"""
Convert the arjunagarwal28/real_kortex_lerobot dataset from LeRobot v3.0 format
to v2.1 format compatible with the installed LeRobot package (0.3.4).

Usage:
    uv run --project envs/smolvla python scripts/convert_kortex_v3_to_v2.py

Reads from:  /scratch/yuchen.yan/smol_vla/datasets/arjunagarwal28/real_kortex_lerobot
Writes to:   /scratch/yuchen.yan/smol_vla/datasets/real_kortex_lerobot_v21
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

V3_ROOT = Path("/scratch/yuchen.yan/smol_vla/datasets/arjunagarwal28/real_kortex_lerobot")
V21_ROOT = Path("/scratch/yuchen.yan/smol_vla/datasets/real_kortex_lerobot_v21")
REPO_ID = "real_kortex_lerobot_v21"


def iter_video_frames(video_path: Path):
    """Yield frames one at a time using imageio's pyav plugin (streaming, low memory)."""
    import imageio.v3 as iio
    for frame in iio.imiter(str(video_path), plugin="pyav"):
        yield frame


def main():
    with open(V3_ROOT / "meta" / "info.json") as f:
        v3_info = json.load(f)

    tasks_df = pd.read_parquet(V3_ROOT / "meta" / "tasks.parquet")
    task_map = dict(zip(tasks_df["task_index"], tasks_df["task"]))

    episodes_df = pd.read_parquet(
        V3_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    data_df = pd.read_parquet(V3_ROOT / "data" / "chunk-000" / "file-000.parquet")
    data_df = data_df.sort_values("index").reset_index(drop=True)

    fps = v3_info["fps"]
    print(f"Dataset: {v3_info['total_episodes']} episodes, {v3_info['total_frames']} frames, {fps} fps")
    print(f"Tasks: {task_map}")

    if V21_ROOT.exists():
        print(f"Removing existing output directory: {V21_ROOT}")
        shutil.rmtree(V21_ROOT)

    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": {
                "state": [
                    "joint_0", "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6",
                    "gripper_0", "gripper_1",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "action": [
                    "delta_eef_0", "delta_eef_1", "delta_eef_2",
                    "delta_eef_3", "delta_eef_4", "delta_eef_5",
                    "gripper",
                ]
            },
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=fps,
        features=features,
        root=V21_ROOT,
        robot_type=v3_info.get("robot_type", "kortex"),
        use_videos=False,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    vid_col = "videos/observation.images.front/file_index"
    vid_chunk_col = "videos/observation.images.front/chunk_index"

    grouped = episodes_df.groupby([vid_chunk_col, vid_col])

    for (chunk_idx, file_idx), group in sorted(grouped):
        group = group.sort_values("episode_index")
        vid_path = (
            V3_ROOT / "videos" / "observation.images.front"
            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        )
        print(f"\nProcessing video: {vid_path.name} ({len(group)} episodes)")

        frame_iter = iter_video_frames(vid_path)
        vid_frame_counter = 0

        for _, ep_row in group.iterrows():
            ep_idx = int(ep_row["episode_index"])
            ep_from = int(ep_row["dataset_from_index"])
            ep_to = int(ep_row["dataset_to_index"])
            ep_length = ep_to - ep_from
            task_idx = int(data_df.iloc[ep_from]["task_index"])
            task_str = task_map.get(task_idx, "unknown task")

            ep_data = data_df.iloc[ep_from:ep_to]

            for local_i, (_, row) in enumerate(ep_data.iterrows()):
                try:
                    img = next(frame_iter)
                    vid_frame_counter += 1
                except StopIteration:
                    print(f"  WARNING: Video ran out of frames at episode {ep_idx}, frame {local_i}")
                    img = np.zeros((256, 256, 3), dtype=np.uint8)

                state = np.array(row["observation.state"], dtype=np.float32)
                action = np.array(row["action"], dtype=np.float32)

                dataset.add_frame(
                    {
                        "observation.images.front": img,
                        "observation.state": state,
                        "action": action,
                    },
                    task=task_str,
                    timestamp=local_i / fps,
                )

            dataset.save_episode()

            if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
                print(f"  Saved episode {ep_idx + 1}/{len(episodes_df)} "
                      f"({ep_length} frames, video frame {vid_frame_counter})")

        print(f"  Finished video {vid_path.name}: {vid_frame_counter} frames consumed")

    dataset.stop_image_writer()

    print(f"\nConversion complete!")
    print(f"v2.1 dataset saved to: {V21_ROOT}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
