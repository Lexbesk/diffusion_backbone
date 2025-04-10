import os
import subprocess


STORE_PATH = '/data/group_data/katefgroup/VLA/voxactb_raw'
LINK = 'https://huggingface.co/datasets/arthur801031/voxact-b/resolve/main/data'

tasks = [
    'hand_over_item',
    'open_drawer',
    'open_jar',
    'put_item_in_drawer'
]

split = 'train'
os.makedirs(f'{STORE_PATH}/{split}', exist_ok=True)
for task in tasks:
    print(task)
    if os.path.exists(f'{STORE_PATH}/{split}/{task}'):
        continue
    cmds = [
        f"wget {LINK}/{split}/{task}_100_demos_corl_v1.zip",
        f"unzip {task}_100_demos_corl_v1.zip",
        f"mv {task}_100_demos_corl_v1/{task} {STORE_PATH}/{split}/",
        f"rm {task}_100_demos_corl_v1.zip",
        f"rm -r {task}_100_demos_corl_v1"
    ]
    for cmd in cmds:
        print(cmd)
        subprocess.run(
            cmd,
            shell=True,
            capture_output=True, text=True, check=True,
            cwd=f"{STORE_PATH}/{split}"
        )

for split in ['val', 'test']:
    os.makedirs(f'{STORE_PATH}/{split}', exist_ok=True)
    for task in tasks:
        print(task)
        if os.path.exists(f'{STORE_PATH}/{split}/{task}'):
            continue
        suf = 'ours_' if task != 'hand_over_item' else ''
        v_ = 'v1' if task != 'put_item_in_drawer' else 'v2'
        cmds = [
            f"wget {LINK}/{split}/{task}_25_demos_{suf}corl_{v_}.zip",
            f"unzip {task}_25_demos_{suf}corl_{v_}.zip",
            f"mv {task}_25_demos_{suf}corl_{v_}/{task} {STORE_PATH}/{split}/",
            f"rm {task}_25_demos_{suf}corl_{v_}.zip",
            f"rm -r {task}_25_demos_{suf}corl_{v_}"
        ]
        for cmd in cmds:
            print(cmd)
            subprocess.run(
                cmd,
                shell=True,
                capture_output=True, text=True, check=True,
                cwd=f"{STORE_PATH}/{split}"
            )
