"""
Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""

import os
import gc
import random
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import tap
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm

from online_evaluation_calvin.evaluate_model import create_model
from online_evaluation_calvin.evaluate_utils import (
    prepare_visual_states,
    prepare_proprio_states,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from online_evaluation_calvin.multistep_sequences import get_sequences
from online_evaluation_calvin.evaluate_utils import get_env

logger = logging.getLogger(__name__)

EP_LEN = 60
NUM_SEQUENCES = 1000
EXECUTE_LEN = 20


class Arguments(tap.Tap):
    # Online environment
    merged_config_file: Path = "online_evaluation_calvin/configs/merged_config_val_abc_d.yaml"
    task_config_file: Path = "online_evaluation_calvin/configs/new_playtable_tasks.yaml"
    ann_config_file: Path = "online_evaluation_calvin/configs/new_playtable_validation.yaml"

    # Eval options
    seed: int = 0
    checkpoint: Path
    base_log_dir: Path = Path(__file__).parent / "eval_logs" / "calvin"
    device: str = "cuda"
    save_video: int = 0

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_attn_heads: int = 8
    num_vis_ins_attn_layers: int = 2
    rotation_parametrization: str = '6D'
    quaternion_format: str = 'wxyz'
    denoise_timesteps: int = 100
    denoise_model: str = 'ddpm'
    fps_subsampling_factor: int = 3
    interpolation_length: int = 2 # the number of steps to reach keypose
    relative_action: int = 0


def evaluate_policy(model, env, task_config_file, ann_config_file,
                    eval_log_dir=None, save_video=False, sequence_indices=[]):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(task_config_file)
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(ann_config_file)

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        result, videos = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video
        )
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip
            clip = []
            for task_ind, (subtask, video) in enumerate(zip(eval_sequence, videos)):
                for img_ind, img in enumerate(video):
                    img = add_text_to_image(img, f'{task_ind}: {subtask}')
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video):
    """
    Evaluates a sequence of language instructions.

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        success, video = rollout(env, model, task_checker,
                                 subtask, lang_annotation)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask, lang_annotation):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name
        lang_annotation: a string indicates the instruction of the task

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = [] # show video for debugging
    obs = env.get_obs()

    model.reset()
    start_info = env.get_info()

    print('------------------------------')
    print(f'task: {lang_annotation}')
    video.append(obs["rgb_obs"]["rgb_static"])

    pbar = tqdm(range(EP_LEN))
    for step in pbar:
        obs = prepare_visual_states(obs, env)
        obs = prepare_proprio_states(obs)
        lang_embeddings = model.encode_instruction(lang_annotation, model.args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory = model.step(obs, lang_embeddings)
        for act_ind in range(min(trajectory.shape[1], EXECUTE_LEN)):
            # calvin_env executes absolute action in the format of:
            # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
            curr_action = [
                trajectory[0, act_ind, :3],
                trajectory[0, act_ind, 3:6],
                trajectory[0, act_ind, [6]]
            ]
            pbar.set_description(f"step: {step}")
            curr_proprio = obs['proprio']
            obs, _, _, current_info = env.step(curr_action)
            obs['proprio'] = curr_proprio

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )

            video.append(obs["rgb_obs"]["rgb_static"])

            if len(current_task_info) > 0:
                return True, video

    return False, video


def add_text_to_image(img, text, position=(10, 180),
                      color=(0, 0, 0)):
    """
    Adds text to a given NumPy image array using PIL.

    Parameters:
    - img (numpy.ndarray): Input image (H, W, C) in uint8 format.
    - text (str): Text to overlay on the image.
    - position (tuple): (x, y) coordinates for the text.
    - font_size (int): Font size of the text.
    - color (tuple): RGB color of the text.

    Returns:
    - numpy.ndarray: Image with text overlay.
    """
    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img)

    # Create a drawing object
    draw = ImageDraw.Draw(pil_img)

    # Load a default font (can replace with custom font using ImageFont.truetype)
    font = ImageFont.load_default()

    # Draw text on the image
    draw.text(position, text, font=font, fill=color)

    # Convert back to NumPy array
    return np.array(pil_img)


def main(args):

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load a custom model wrapper
    model = create_model(args)

    # Split sequence indices for multi-processing
    sequence_indices = [
        i for i in range(args.local_rank, NUM_SEQUENCES, int(os.environ["WORLD_SIZE"]))
    ]

    # Make environment
    env = get_env(args.merged_config_file, show_gui=False)

    # Run evaluation on all episodes
    evaluate_policy(model, env,
                    task_config_file=args.task_config_file,
                    ann_config_file=args.ann_config_file,
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video)

    # Gather statistics from the whole dataset
    results, sequence_inds = collect_results(args.base_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f'Load {len(results)}/1000 episodes...')
    print(str_results + "\n")

    del env
    gc.collect()


if __name__ == "__main__":
    args = Arguments().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
