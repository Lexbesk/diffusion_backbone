import os
from subprocess import call
import pickle
from pathlib import Path
import shutil

import tap


class Arguments(tap.Tap):
    root_dir: Path


def main_old(root_dir, task):
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
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

        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            call(['ln', '-s',
                  f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl',
                  f'{root_dir}/{task}/variation{variation}/'])

        ep_id = len(seen_variations[variation]) - 1
        call(['ln', '-s',
              "{:s}/{:s}/all_variations/episodes/episode{:d}".format(root_dir, task, num),
              f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'])


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
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    for task in tasks:
        main(root_dir, task)
