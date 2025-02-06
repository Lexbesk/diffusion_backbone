import json
import os


FOLDER = 'eval_logs/flow_flow_3dda_gnfactor/seed0/'

sum_ = 0
tasks = os.listdir(FOLDER)
for folder in tasks:
    with open(f'{FOLDER}/{folder}/eval.json') as fid:
        res = json.load(fid)[folder]["mean"]
    print(folder, res)
    sum_ += res
print(f'Mean on {len(tasks)} tasks', sum_ / len(tasks))
