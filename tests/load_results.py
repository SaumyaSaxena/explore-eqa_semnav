import pickle, json
import numpy as np

if __name__ == "__main__":

    # Replace 'your_file.pkl' with the path to your .pkl file
    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/vlm_exp_semantic/'
    with open(results_path + 'results.pkl', 'rb') as file:
        results = pickle.load(file)
    filename = results_path + 'metrics2.json'
    # data keys are ['question_ind', 'step_0', 'step_1', 'step_2', 'step_3', 'step_4', 'step_5', 'step_6', 'step_7', 'step_8', 'step_9', 'step_10', 'step_11', 'step_12', 'step_13', 'step_14', 'step_15', 'step_16', 'step_17', 'step_18', 'step_19', 'step_20', 'step_21', 'step_22', 'step_23', 'step_24', 'step_25', 'step_26', 'step_27', 'step_28', 'step_29', 'step_30', 'step_31', 'step_32', 'step_33', 'step_34']
    
    metrics = {}
    total_length_all_trajs = 0.
    weighted_length_all_trajs = 0.
    max_length_all_trajs = 0.
    planning_steps_weighted_all_trajs = 0
    planning_steps_max_all_trajs = 0
    planning_steps_total = 0
    for i in range(len(results)):
        result = results[i]
        steps = [k.split('step_')[1] for k in result.keys() if k.startswith('step')]
        num_steps = len(steps)
        planning_steps_total += num_steps
        pts = np.array([result[f'step_{s}']['pts'] for s in steps])
        deltas = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        total_length = np.sum(segment_lengths)
        total_length_all_trajs += total_length
        

        #best weighted index
        smx_vlm_pred = np.array([result[f'step_{s}']['smx_vlm_pred'] for s in steps])
        smx_vlm_rel = np.array([result[f'step_{s}']['smx_vlm_rel'][0] for s in steps])
        smx_vlm_weighted = smx_vlm_pred*smx_vlm_rel[:,np.newaxis]
        smx_weighted_max_idx = np.argmax(smx_vlm_weighted, axis=0)
        smx_weighted_max = np.max(smx_vlm_weighted, axis=0)
        best_weighted_idx = smx_weighted_max_idx[np.argmax(smx_weighted_max)]
        planning_steps_weighted_all_trajs += best_weighted_idx
        # Traj len
        weighted_length_all_trajs += np.sum(segment_lengths[:best_weighted_idx])

        #best max index
        best_max_idx = np.argmax(smx_vlm_rel)
        planning_steps_max_all_trajs += best_max_idx
        # Traj len
        max_length_all_trajs += np.sum(segment_lengths[:best_max_idx])

        metrics[f'index_{i}'] = {
            'length_traj': total_length,
            'num_total_planning_steps': float(num_steps),
            'best_weighted_step': float(best_weighted_idx),
            'best_max_step': float(best_max_idx)
        }
        
    metrics['total_length_all_trajs'] = total_length_all_trajs
    metrics['weighted_length_all_trajs'] = weighted_length_all_trajs
    metrics['max_length_all_trajs'] = max_length_all_trajs
    metrics['planning_steps_weighted_all_trajs'] = float(planning_steps_weighted_all_trajs)
    metrics['planning_steps_max_all_trajs'] = float(planning_steps_max_all_trajs)
    metrics['num_episodes'] = len(results)
    metrics['planning_steps_total'] = planning_steps_total

    print(f"Saving file: {filename}")
    with open(filename, 'w') as file:
        json.dump(metrics, file, indent=4)
    print(f"Saved file: {filename}")

