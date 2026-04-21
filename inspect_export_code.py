import json
import os

import h5py
import numpy as np


def save_demonstration_as_hdf5(new_dir, i, data, task_description, task_nouns):
    """
    Save one demonstration episode as:
        <new_dir>/demo_<i>.hdf5
        <new_dir>/demo_<i>_metainfo.json
    """

    initial_state = data["initial_state"]
    initial_state_seed = data["initial_state_seed"]

    actions = np.asarray(data["actions"])
    states = data["states"]
    rewards = data["rewards"]
    gripper_states = data["gripper_states"]
    joint_states = data["joint_states"]
    robot_states = data["robot_states"]
    ee_states = data["ee_states"]

    agentview_images = data["agentview_images"]
    agentview_segs = data["agentview_segs"]
    agentview_contacts = data["agentview_contacts"]
    agentview_boxes = data["agentview_boxes"]

    T = len(actions)

    required_sequences = {
        "states": states,
        "gripper_states": gripper_states,
        "joint_states": joint_states,
        "robot_states": robot_states,
        "ee_states": ee_states,
        "agentview_images": agentview_images,
        "agentview_segs": agentview_segs,
        "agentview_contacts": agentview_contacts,
    }

    for name, seq in required_sequences.items():
        if len(seq) != T:
            raise ValueError(
                f"Length mismatch: actions has length {T}, but {name} has length {len(seq)}"
            )

    dones = np.zeros(T, dtype=np.uint8)
    dones[-1] = 1

    rewards4done = rewards["done"]
    rewards4dist = rewards["distance"]
    rewards4overlap = rewards["overlap"]

    states_arr = np.stack(states, axis=0)
    gripper_states_arr = np.stack(gripper_states, axis=0)
    joint_states_arr = np.stack(joint_states, axis=0)
    robot_states_arr = np.stack(robot_states, axis=0)
    ee_states_arr = np.stack(ee_states, axis=0)
    agentview_rgb_arr = np.stack(agentview_images, axis=0)
    agentview_seg_arr = np.stack(agentview_segs, axis=0)
    agentview_contact_arr = np.stack(agentview_contacts, axis=0)

    if ee_states_arr.shape[-1] < 6:
        raise ValueError(
            f"ee_states should have last dim >= 6, got shape {ee_states_arr.shape}"
        )

    ee_pos_arr = ee_states_arr[:, :3]
    ee_ori_arr = ee_states_arr[:, 3:]

    os.makedirs(new_dir, exist_ok=True)
    hdf5_path = os.path.join(new_dir, f"demo_{i}.hdf5")

    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        ep_data_grp = grp.create_group(f"demo_{i}")
        obs_grp = ep_data_grp.create_group("obs")

        obs_grp.create_dataset("states", data=states_arr)
        obs_grp.create_dataset("gripper_states", data=gripper_states_arr)
        obs_grp.create_dataset("joint_states", data=joint_states_arr)
        obs_grp.create_dataset("ee_states", data=ee_states_arr)
        obs_grp.create_dataset("ee_pos", data=ee_pos_arr)
        obs_grp.create_dataset("ee_ori", data=ee_ori_arr)

        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb_arr)
        obs_grp.create_dataset("agentview_seg", data=agentview_seg_arr)
        obs_grp.create_dataset("agentview_contact", data=agentview_contact_arr)

        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("robot_states", data=robot_states_arr)
        ep_data_grp.create_dataset("rewards4done", data=rewards4done)
        ep_data_grp.create_dataset("rewards4dist", data=rewards4dist)
        ep_data_grp.create_dataset("rewards4overlap", data=rewards4overlap)
        ep_data_grp.create_dataset("dones", data=dones)

        ep_data_grp.attrs["num_samples"] = T

    episode_key = i
    task_key = task_description.replace(" ", "_")
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(new_dir, f"demo_{i}_metainfo.json")

    if task_key not in metainfo_json_dict:
        metainfo_json_dict[task_key] = {}
    if episode_key not in metainfo_json_dict[task_key]:
        metainfo_json_dict[task_key][episode_key] = {}

    metainfo_json_dict[task_key][episode_key]["success"] = True
    metainfo_json_dict[task_key][episode_key]["initial_state"] = initial_state.tolist()
    metainfo_json_dict[task_key][episode_key]["initial_state_seed"] = initial_state_seed
    metainfo_json_dict[task_key][episode_key]["task_nouns"] = task_nouns
    metainfo_json_dict[task_key][episode_key]["task_description"] = task_description
    metainfo_json_dict[task_key][episode_key]["exo_boxes"] = agentview_boxes

    with open(metainfo_json_out_path, "w") as f:
        json.dump(metainfo_json_dict, f, indent=2)

    return hdf5_path
