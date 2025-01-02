"""Online evaluation script on RLBench."""
import random
from pathlib import Path
import json
import os
import pickle

import torch
import numpy as np
import argparse

from diffuser_actor.policy.trajectory_optimization.denoise_actor import DenoiseActor
from utils.utils_with_rlbench import RLBenchEnv, Actioner, load_episodes
from utils.common_utils import str2bool, str_none, round_floats
from datasets.dataset_rlbench import (
    GNFactorDataset,
    PeractDataset,
    Peract2Dataset
)


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for evaluate_policy.py")
    # Trainign and testing
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=str_none, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--image_size', type=str, default="256,256")
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--headless', type=str2bool, default=False)
    parser.add_argument('--predict_trajectory', type=str2bool, default=False)
    parser.add_argument('--max_tries', type=int, default=10)
    parser.add_argument('--dataset', type=str, default="Peract")
    parser.add_argument('--instructions', type=str, default="instructions.pkl")
    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parent / "demos"))
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--output_file', type=str, default=str(Path(__file__).parent / "eval.json"))
    parser.add_argument('--max_steps', type=int, default=25)
    parser.add_argument('--collision_checking', type=str2bool, default=False)
    parser.add_argument('--denoise_timesteps', type=int, default=10)
    parser.add_argument('--denoise_model', type=str, default="rectified_flow",
                        choices=["ddpm", "rectified_flow"])
    parser.add_argument('--num_history', type=int, default=1)
    parser.add_argument('--fps_subsampling_factor', type=int, default=5)
    parser.add_argument('--dense_interpolation', type=str2bool, default=False)
    parser.add_argument('--interpolation_length', type=int, default=100)
    parser.add_argument('--relative_action', type=str2bool, default=False)
    parser.add_argument('--action_dim', type=int, default=8)
    parser.add_argument('--backbone', type=str, default="clip")
    parser.add_argument('--embedding_dim', type=int, default=120)
    parser.add_argument('--num_attn_heads', type=int, default=9)
    parser.add_argument('--num_vis_ins_attn_layers', type=int, default=2)
    parser.add_argument('--use_instruction', type=int, default=1)
    parser.add_argument('--rotation_parametrization', type=str, default='quat')
    parser.add_argument('--quaternion_format', type=str, default='wxyz')

    return parser.parse_args()


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    model = DenoiseActor(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
        use_instruction=args.use_instruction,
        fps_subsampling_factor=args.fps_subsampling_factor,
        rotation_parametrization=args.rotation_parametrization,
        quaternion_format=args.quaternion_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        nhist=args.num_history,
        relative=args.relative_action,
    )

    # Load model weights
    model_dict = torch.load(args.checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)
    model.eval()

    return model


if __name__ == "__main__":
    # Arguments
    args = parse_arguments()
    dataset_cls = {
        "Peract": PeractDataset,
        "Peract2": Peract2Dataset,
        "GNFactor": GNFactorDataset
    }[args.dataset]

    print("Arguments:")
    print(args)
    print("-" * 100)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=dataset_cls.cameras,
        collision_checking=bool(args.collision_checking)
    )

    with open(args.instructions, "rb") as fid:
        instruction = pickle.load(fid)

    actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=dataset_cls.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory)
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in dataset_cls.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=dataset_cls.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            max_tries=args.max_tries,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
