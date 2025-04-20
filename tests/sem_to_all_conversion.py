import pickle, json
import numpy as np
import csv, os

if __name__ == "__main__":

    # Load dataset
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/data/questions.csv') as f:
        full_questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    # Filter to include only scenes with semantic annotations
    semantic_annot_data_path = '/home/saumyas/catkin_ws_semnav/data/hm3d-train-semantic-annots-v0.2'
    semantic_scenes = [f for f in os.listdir(semantic_annot_data_path) if os.path.isdir(os.path.join(semantic_annot_data_path, f))]

    questions_data = []
    count = 0
    sem_to_all_indx = {}
    all_to_sem_indx = {}
    for i in range(len(full_questions_data)):
        data = full_questions_data[i]
        if data['scene'] in semantic_scenes:
            questions_data.append(data)
            scene_floor = data["scene"] + "_" + data["floor"]
            experiment_id = f'{count}_{data["scene"]}_{data["floor"]}'

            sem_to_all_indx[count] = {'all_idx': i, 'label': experiment_id}
            all_to_sem_indx[i] = count
            count += 1

    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/results/sem_to_all_indx.json', 'w') as file:
        json.dump(sem_to_all_indx, file, indent=4)
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/results/all_to_sem_indx.json', 'w') as file:
        json.dump(all_to_sem_indx, file, indent=4)

    # for question_ind in range(len(questions_data)):
    #     import ipdb; ipdb.set_trace()
