import json
import numpy as np


def check_if_multifloor(traj):
    mean = sum(traj) / traj.shape[0]
    variance = sum((traj - mean)**2) / traj.shape[0]
    return np.sqrt(variance) > 0.5

def load_openeqa_data(cfg):
    # Load dataset
    with open(cfg.openeqa_question_data_path, "r") as file:
        questions_data = json.load(file)
    
    with open(cfg.semantic_annot_data_path, "r") as file:
        semantic_annots = json.load(file)
    semantic_annots = semantic_annots['stages']['paths']['.glb']
    
    with open(cfg.openeqa_init_pose_data_path, "r") as file:
        init_poses = json.load(file)
    
    with open(cfg.openeqa_choices_data_path, "r") as file:
        choices = json.load(file)


    semantic_scenes = [s.split('/*.basis')[0] for s in semantic_annots]

    filtered_question_data = []
    for data in questions_data:
        if 'hm3d-v0' in data['episode_history']:
            scene_id = init_poses[data['episode_history']]['scene_id'].split('/')[0]
            data['scene'] = scene_id
            if cfg.use_only_semantic_data:
                if scene_id in semantic_scenes:
                    if cfg.use_multifloor_questions:
                        filtered_question_data.append(data)
                    else:
                        if not check_if_multifloor(np.array(init_poses[data['episode_history']]['full_traj_pos'])[:,1]):
                            filtered_question_data.append(data)
            else:
                if scene_id not in semantic_scenes:
                    if cfg.use_multifloor_questions:
                        filtered_question_data.append(data)
                    else:
                        if not check_if_multifloor(np.array(init_poses[data['episode_history']]['full_traj_pos'])[:,1]):
                            filtered_question_data.append(data)

    print(f"Loaded {len(filtered_question_data)} questions.")
    return filtered_question_data, init_poses, choices