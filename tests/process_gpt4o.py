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
    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gpt4o_early_term/'
    good_pkls = [results_path+"results_130.pkl", results_path+"results_140.pkl", results_path+"results_170.pkl",results_path+"results_190.pkl", results_path+"results_200.pkl", results_path+"results_210.pkl", results_path+"results_250.pkl", results_path+"results_300.pkl", results_path+"results_310.pkl",results_path+"results_320.pkl",results_path+"results_340.pkl", results_path+"results_350.pkl"] 

    
    filename = results_path + 'metrics_succ_conf_all_ques.json'
    filename_task = results_path + 'task_category_conf_all_ques.json'

    metrics = {}
    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_succ_weighted = 0
    num_succ_max = 0

    ques_count = 0
    
    for file_name in good_pkls:
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
        
        q_indxs = [result['question_ind'] for result in results]
        print(f"Question indexes in file {file}")
        print(q_indxs)
        # num_episodes += len(results)

        for result in results:
            if ques_count>319 and ques_count<330:
                ques_count+=1
                continue
            if ques_count == result['question_ind']:
                metrics[result['question_ind']] = {}
                metrics[result['question_ind']]['success_max'] = False
                metrics[result['question_ind']]['success_at_step0'] = False

                print(f"question idx:{ques_count}")
                ques_count+=1
                num_weighted_steps = result['num_weighted_steps']
                num_max_steps = result['num_max_steps']
                if result['success_weighted'] and result[f'step_{num_weighted_steps}']['smx_vlm_rel'][0]>0.5:
                    weighted_length_all_trajs += result['weighted_traj_len']
                    planning_steps_weighted_all_trajs += result['num_weighted_steps']
                    num_succ_weighted += 1
                
                if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                    max_length_all_trajs += result['max_traj_len']
                    planning_steps_max_all_trajs += result['num_max_steps']
                    num_succ_max += 1
                    metrics[result['question_ind']]['success_max'] = True
                    if result['num_max_steps'] == 0:
                        metrics[result['question_ind']]['success_at_step0'] = True
                        # print(result['question_ind'])

                if full_questions_data[result['question_ind']]['label'] == 'identification':
                    identification += 1
                    if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        identification_succ += 1
                elif full_questions_data[result['question_ind']]['label'] == 'existence':
                    existence+=1
                    if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        existence_succ += 1
                elif full_questions_data[result['question_ind']]['label'] == 'count':
                    count+=1
                    if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        count_succ += 1
                elif full_questions_data[result['question_ind']]['label'] == 'state':
                    state+=1
                    if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        state_succ += 1
                elif full_questions_data[result['question_ind']]['label'] == 'location':
                    location+=1
                    if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        location_succ += 1
                else:
                    raise NotImplementedError("invalid question type")


    metrics['weighted_length_all_trajs'] = weighted_length_all_trajs
    metrics['max_length_all_trajs'] = max_length_all_trajs
    metrics['planning_steps_weighted_all_trajs'] = float(planning_steps_weighted_all_trajs)
    metrics['planning_steps_max_all_trajs'] = float(planning_steps_max_all_trajs)
    metrics['num_episodes'] = ques_count
    metrics['num_succ_weighted'] = float(num_succ_weighted)
    metrics['num_succ_max'] = float(num_succ_max)

    print(f"Saving file: {filename}")
    with open(filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {filename}")

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