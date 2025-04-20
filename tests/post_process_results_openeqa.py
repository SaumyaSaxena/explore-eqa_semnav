import pickle, json
import numpy as np
import csv, os

from omegaconf import OmegaConf

from src.utils import load_openeqa_data

if __name__ == "__main__":

    cfg_file = '/home/saumyas/Projects/semnav/explore-eqa_semnav/cfg/openeqa_prismatic_exp.yaml'
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)

    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/openeqa_gpt4o_early_term/'
    metrics_filename = results_path + 'metrics.json'
    questions_data, init_pose_data, choices_data = load_openeqa_data(cfg)

    type_results = {}
    type_results['spatial understanding'] = 0
    type_results['object state recognition'] = 0
    type_results['functional reasoning'] = 0
    type_results['attribute recognition'] = 0
    type_results['world knowledge'] = 0
    type_results['object localization'] = 0
    type_results['object recognition'] = 0

    all_categories = ['spatial understanding', 'object state recognition', 'functional reasoning', 'attribute recognition', 'world knowledge', 'object localization', 'object recognition']
    for category in all_categories:
        type_results[f'{category}'] = 0
        type_results[f'{category} success max'] = 0
        type_results[f'{category} success weighted'] = 0

    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_succ_weighted = 0
    num_succ_max = 0

    result_files = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f)) and f.endswith('.pkl')]
    result_files = [os.path.join(results_path, f) for f in result_files]

    ques_ids_processed = []
    for result_file in result_files:
        with open(result_file, 'rb') as file:
            results = pickle.load(file)

        print(f"Processing file: {result_file}")
        for result in results:
            question_ind = result['question_ind']
            question_id = result['question_id']
            if question_id in ques_ids_processed:
                print(f"Already processed question_ind: {question_ind}")
                continue
            print(f'{question_ind=}')
            ques_ids_processed.append(question_id)

            if 'category' in result:
                category = result['category']
            else:
                question_data = questions_data[question_ind]
                category = question_data["category"]

            type_results[category] += 1

            if result['success_weighted']:
                weighted_length_all_trajs += result['weighted_traj_len']
                planning_steps_weighted_all_trajs += result['num_weighted_steps']
                num_succ_weighted += 1
                type_results[f'{category} success weighted'] += 1
            if result['success_max']:
                max_length_all_trajs += result['max_traj_len']
                planning_steps_max_all_trajs += result['num_max_steps']
                num_succ_max += 1
                type_results[f'{category} success max'] += 1
    
    
    metrics = {}
    metrics['avg_weighted_length_all_trajs'] = float(weighted_length_all_trajs/num_succ_weighted)
    metrics['avg_max_length_all_trajs'] = float(max_length_all_trajs/num_succ_max)
    metrics['avg_planning_steps_weighted_all_trajs'] = float(planning_steps_weighted_all_trajs/num_succ_weighted)
    metrics['avg_planning_steps_max_all_trajs'] = float(planning_steps_max_all_trajs/num_succ_max)
    metrics['num_succ_weighted'] = float(num_succ_weighted)
    metrics['num_succ_max'] = float(num_succ_max)
    metrics['percent_succ_weighted'] = float(num_succ_weighted/len(ques_ids_processed))
    metrics['percent_succ_max'] = float(num_succ_max/len(ques_ids_processed))
    metrics['num_episodes'] = len(ques_ids_processed)

    for category in all_categories:
        type_results[f'{category} success max percent'] = type_results[f'{category} success max']/type_results[category]*100
        type_results[f'{category} success weighted percent'] = type_results[f'{category} success weighted']/type_results[category]*100

    metrics['type_results'] = type_results

    print(f"Saving file: {metrics_filename}")
    with open(metrics_filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {metrics_filename}")