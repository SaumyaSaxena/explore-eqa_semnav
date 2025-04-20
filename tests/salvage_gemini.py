import pickle, json, os
import numpy as np
from omegaconf import OmegaConf
import csv

if __name__ == "__main__":

    # Load dataset
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/data/questions.csv') as f:
        full_questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    # Replace 'your_file.pkl' with the path to your .pkl file
    results_path1 = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gemini_all_data/'
    results_path2 = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gemini_all_data_contd/'
    results_path3 = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gemini_all_data_contd_early_term/'

    filename = results_path3 + 'metrics.json'
    metrics = {}
    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_episodes = 0
    num_succ_weighted = 0
    num_succ_max = 0
    good_pkls = [results_path1+"results_150.pkl", results_path1+"results_40.pkl", results_path2+"results_220.pkl", results_path3+"results.pkl"] 
    
    ques_count = 0
    for file_name in good_pkls:
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
        
        q_indxs = [result['question_ind'] for result in results]
        print(f"Question indexes in file {file}")
        print(q_indxs)

        # num_episodes += len(results)
        for result in results:
            if ques_count == result['question_ind']:
                print(f"question idx:{ques_count}")
                ques_count+=1
                if result['success_weighted']:
                    weighted_length_all_trajs += result['weighted_traj_len']
                    planning_steps_weighted_all_trajs += result['num_weighted_steps']
                    num_succ_weighted += 1
                if result['success_max']:
                    max_length_all_trajs += result['max_traj_len']
                    planning_steps_max_all_trajs += result['num_max_steps']
                    num_succ_max += 1
        
    metrics['weighted_length_all_trajs'] = weighted_length_all_trajs
    metrics['max_length_all_trajs'] = max_length_all_trajs
    metrics['planning_steps_weighted_all_trajs'] = float(planning_steps_weighted_all_trajs)
    metrics['planning_steps_max_all_trajs'] = float(planning_steps_max_all_trajs)
    metrics['num_episodes'] = num_episodes
    metrics['num_succ_weighted'] = float(num_succ_weighted)
    metrics['num_succ_max'] = float(num_succ_max)

    print(f"Saving file: {filename}")
    with open(filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {filename}")