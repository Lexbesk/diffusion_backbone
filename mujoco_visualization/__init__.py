import os
import multiprocessing
import logging
from glob import glob
import traceback

import numpy as np
import torch
from .eval_func import fcMocapEval
from types import SimpleNamespace

def dict_to_namespace(obj):
    """Recursively convert dictionaries (and lists of dictionaries)
    into SimpleNamespace so we can use dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


# ---------- raw config as a plain dict ----------
raw_cfg = {
    # "skip": True,
    # "n_worker": 48,
    # "setting": "fc",
    # "exp_name": "dexonomy",
    # "save_root": "output",
    # "save_dir": "${save_root}/${exp_name}_${hand_name}",
    # "grasp_dir": "${save_dir}/graspdata",
    # "eval_dir": "${save_dir}/evaluation",
    # "succ_dir": "${save_dir}/succgrasp",
    # "collect_dir": "${save_dir}/succ_collect",
    # "vusd_dir": "${save_dir}/vis_usd",
    # "vobj_dir": "${save_dir}/vis_obj",
    # "log_dir": "${save_dir}/log/${task_name}/${now:%Y_%m_%d_%H_%M_%S}",
    # "task_name": "${hydra:runtime.choices.task}",
    # "hand_name": "${hydra:runtime.choices.hand}",

    "task": {
        "max_num": 1000,
        "obj_mass": 0.1,
        "miu_coef": [0.6, 0.02],
        "debug_render": True,
        "debug_viewer": False,
        "debug_dir": "${save_dir}/debug",
        "simulation_test": True,

        "simulation_metrics": {
            "max_pene": 0.01,
            "trans_thre": 0.05,
            "angle_thre": 90,
        },
        "analytic_fc_metrics": {
            "contact_tip_only": True,
            "contact_threshold": 0.005,
            "type": ["qp", "qp_dfc", "q1", "tdg", "dfc"],
        },
        "pene_contact_metrics": {
            "contact_margin": 0.01,
            "contact_threshold": 0.002,
        },
    },

    "hand": {
        "xml_path": "/data/user_data/austinz/Robots/DexGraspBench/assets/hand/shadow/right_hand.xml",
        "mocap": True,
        "exclude_table_contact": None,
        "color": [0.863157, 0.0865002, 0.0802199, 1.0],
        "finger_prefix": ["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"],
        "valid_body_name": [
            "rh_palm", "rh_ffproximal", "rh_ffmiddle", "rh_ffdistal",
            "rh_mfproximal", "rh_mfmiddle", "rh_mfdistal",
            "rh_rfproximal", "rh_rfmiddle", "rh_rfdistal",
            "rh_lfmetacarpal", "rh_lfproximal", "rh_lfmiddle", "rh_lfdistal",
            "rh_thproximal", "rh_thmiddle", "rh_thdistal",
        ],
        "tip_body_name": [
            "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal",
        ],
    },
}

# ---------- convert to namespace ----------
CONFIGS = dict_to_namespace(raw_cfg)




def safe_eval_one(params, grasp_data=None):
    input_npy_path, configs, num, visualize = params[0], params[1], params[2], params[3]
    # print(f"Eval {input_npy_path}")
    try:
        results = fcMocapEval(input_npy_path, configs, num, visualize).run()
        succ = results["succ_flag"]
        delta_pos = results["delta_pos"]
        delta_angle = results["delta_angle"]
        return succ
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return
    
COUNT = 0
def val_batch(batch, vis_path): # numpy batch
    global CONFIGS
    configs = CONFIGS
    configs.vis_path = vis_path
    batch_len = batch['grasp_qpos'].shape[0]
    success_count = 0
    failure_count = 0
    for j in range(batch_len):
        global COUNT
        COUNT += 1
        grasp_data = {}
        for key, val in batch.items():
            if key == 'partial_points':
                item = val[j][:1, :]
            else:
                item = val[j]
            grasp_data[key] = item  
        visualize = True if COUNT % 100 == 0 else False
        ip = (grasp_data, configs, f'sample_grasp_{COUNT}', visualize)
        succ = safe_eval_one(ip)
        if succ:
            # print(f"Grasp {j} succeeded")
            success_count += 1
        else:
            # print(f"Grasp {j} failed")
            failure_count += 1
            
    print(f"Success count: {success_count}, Failure count: {failure_count}")
    print(f"accuracy: {success_count / (success_count + failure_count)}")
    return success_count, failure_count