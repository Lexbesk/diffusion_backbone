import argparse
import os
import pickle
from pathlib import Path
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    parser.add_argument('--root_dir', type=Path, required=True)
    return parser.parse_args()


def main(root_dir, task):
    variations = [
        fname
        for fname in os.listdir(f'{root_dir}/{task}/all_variations/episodes')
        if not fname.startswith('.DS')
    ]
    seen_variations = {}
    for variation in variations:
        num = int(variation.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        ep_id = len(seen_variations[variation]) - 1
        folder = f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'
        os.makedirs(folder, exist_ok=True)
        shutil.copy(
            f"{root_dir}/{task}/all_variations/episodes/episode{num}/low_dim_obs.pkl",
            f"{folder}/low_dim_obs.pkl"
        )


if __name__ == '__main__':
    args = parse_arguments()
    root_dir = str(args.root_dir.absolute())
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    for task in tasks:
        main(root_dir, task)
