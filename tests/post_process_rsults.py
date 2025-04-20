import pickle, json
import numpy as np
import csv

if __name__ == "__main__":

    # Load dataset
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/data/questions.csv') as f:
        full_questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    identification, existence, count, state, location = 0, 0, 0, 0, 0
    identification_succ, existence_succ, count_succ, state_succ, location_succ = 0, 0, 0, 0, 0

    # Replace 'your_file.pkl' with the path to your .pkl file
    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gpt4o_one_step/'
    with open(results_path + 'results.pkl', 'rb') as file:
        results = pickle.load(file)
    filename = results_path + 'metrics_succ_new.json'
    filename_task = results_path + 'task_category.json'

    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_succ_weighted = 0
    num_succ_max = 0

    for i in range(len(results)):
        result = results[i]
        if result['success_weighted']:
            weighted_length_all_trajs += result['weighted_traj_len']
            planning_steps_weighted_all_trajs += result['num_weighted_steps']
            num_succ_weighted += 1
        if result['success_max']:
            max_length_all_trajs += result['max_traj_len']
            planning_steps_max_all_trajs += result['num_max_steps']
            num_succ_max += 1

        if full_questions_data[result['question_ind']]['label'] == 'identification':
            identification += 1
            if result['success_max']:
                identification_succ += 1
        elif full_questions_data[result['question_ind']]['label'] == 'existence':
            existence+=1
            if result['success_max']:
                existence_succ += 1
        elif full_questions_data[result['question_ind']]['label'] == 'count':
            count+=1
            if result['success_max']:
                count_succ += 1
        elif full_questions_data[result['question_ind']]['label'] == 'state':
            state+=1
            if result['success_max']:
                state_succ += 1
        elif full_questions_data[result['question_ind']]['label'] == 'location':
            location+=1
            if result['success_max']:
                location_succ += 1
        else:
            raise NotImplementedError("invalid question type")
    
    
    metrics = {}
    metrics['weighted_length_all_trajs'] = weighted_length_all_trajs
    metrics['max_length_all_trajs'] = max_length_all_trajs
    metrics['planning_steps_weighted_all_trajs'] = float(planning_steps_weighted_all_trajs)
    metrics['planning_steps_max_all_trajs'] = float(planning_steps_max_all_trajs)
    metrics['num_succ_weighted'] = float(num_succ_weighted)
    metrics['num_succ_max'] = float(num_succ_max)
    metrics['num_episodes'] = len(results)


    type_results = {}
    type_results['identification'] = identification
    type_results['existence'] = existence
    type_results['count'] = count
    type_results['state'] = state
    type_results['location'] = location
    type_results['total_trajs'] = len(results)

    type_results['identification_succ'] = identification_succ/identification*100
    type_results['existence_succ'] = existence_succ/existence*100
    type_results['count_succ'] = count_succ/count*100
    type_results['state_succ'] = state_succ/state*100
    type_results['location_succ'] = location_succ/location*100

    print(f"Saving file: {filename_task}")
    with open(filename_task, 'w') as file:
        json.dump(type_results, file, indent=4)
    print(f"Saved file: {filename_task}")

    print(f"Saving file: {filename}")
    with open(filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {filename}")