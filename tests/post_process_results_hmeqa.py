import pickle, json
import numpy as np
import csv

def get_num_steps(result_keys):
    step_count = 0
    for key in result_keys:
        if 'step_' in key:
            step_count+=1

    return step_count

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
    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gemini_early_term_non_semantic_only/'
    # Good Gemini pickles
    # good_pkls = [results_path+"results_10.pkl", results_path+"results_20.pkl", results_path+"results_30.pkl", results_path+"results_40.pkl",
    #             results_path+"results_50.pkl", results_path+"results_100.pkl", results_path+"results_110.pkl"] 

    # good llama pickles
    # good_pkls = [results_path+"results_10.pkl",
    #             results_path+"results_70.pkl", results_path+"results_80.pkl",
    #              results_path+"results_90.pkl", results_path+"results_100.pkl", results_path+"results_110.pkl"] 

    # good llama non semantic only pickles
    good_pkls = [results_path+"results.pkl"]
    good_pkls = []
    for i in range(1, 34):
        if i in [3, 7, 9, 11, 13, 17, 23, 27, 31, 33]:
            good_pkls.append(results_path + "results_" + str(i) + "0.pkl")
    # good Gemini pickles


    filename = results_path + 'metrics_succ_conf_all_ques.json'
    filename_task = results_path + 'task_category_conf_all_ques.json'

    metrics = {}
    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    num_succ_weighted = 0
    num_succ_max = 0
    total_steps = 0
    total_traj_len = 0

    ques_count = 0
    
    for file_name in good_pkls:
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
        
        # import ipdb; ipdb.set_trace()
        q_indxs = [result['question_ind'] for result in results]
        print(f"Question indices in file {file}")
        print(q_indxs)
        # num_episodes += len(results)

        for result in results:
            
            # Here is where we reset the ques_count if we are missing a few experiments
            # Llama: missing entries 10, 11, 100
            # if result['question_ind'] == 12:
            #     ques_count = result['question_ind']
            # if result['question_ind'] == 101:
            #     ques_count = result['question_ind']

            # Gemini: missing entries 50 - 55, 100
            if result['question_ind'] == 39:
                ques_count = result['question_ind']

            # import ipdb; ipdb.set_trace()
            if ques_count == result['question_ind']:
                metrics[result['question_ind']] = {}
                metrics[result['question_ind']]['success_max'] = False
                metrics[result['question_ind']]['success_at_step0'] = False

                print(f"question count:{ques_count}, question index:{result['question_ind']}")
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

                total_steps += get_num_steps(result.keys())
                total_traj_len += result['total_traj_len']

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
    metrics['avg_num_max_steps'] = float(planning_steps_max_all_trajs / total_steps)
    metrics['avg_num_weighted_steps'] = float(planning_steps_weighted_all_trajs / total_steps)
    metrics['avg_traj_len_weighted'] = float(weighted_length_all_trajs / total_traj_len)
    metrics['avg_traj_len_max'] = float(max_length_all_trajs / total_traj_len)

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
    if location != 0:
        type_results['location_succ'] = location_succ/location*100

    print(f"Saving file: {filename_task}")
    with open(filename_task, 'w') as file:
        json.dump(type_results, file, indent=4)
    print(f"Saved file: {filename_task}")