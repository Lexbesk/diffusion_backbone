import json
import os


FOLDER = 'eval_logs/refactored_better_3dda_peract_sc20/ablate_moritz.pth/seed0'

sum_ = 0
tasks = sorted(os.listdir(FOLDER))
results = []
for folder in tasks:
    with open(f'{FOLDER}/{folder}/eval.json') as fid:
        res = 100 * json.load(fid)[folder]["mean"]
    results.append(res)
    print(folder, res)
    sum_ += res
print(f'Mean on {len(tasks)} tasks', sum_ / len(tasks))
