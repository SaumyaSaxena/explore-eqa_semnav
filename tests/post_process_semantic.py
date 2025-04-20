import pickle, json
import numpy as np
import csv, os, time
import gspread
from tqdm import trange

def load_eqa_data():
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
    for data in full_questions_data:
        if data['scene'] in semantic_scenes:
            questions_data.append(data)

    return questions_data

def load_spreadsheet(method='Explore-EQA'):
    gc = gspread.service_account('/home/saumyas/.config/gspread/saumya-default-project-8b699ae84be9.json')
    sh = gc.open("GraphEQA results 2024")
    worksheet = sh.worksheet("Sheet4")
    method_col = worksheet.find(method, in_row=1).col
    return worksheet, method_col

def get_good_pickles(method='Explore-EQA'):
    if method == 'Explore-EQA' or method == 'Explore-EQA_steps':
        results_paths = ['/home/saumyas/Projects/semnav/explore-eqa_semnav/results/prismatic_all_data/results.pkl']
        return results_paths

    if method == 'Explore-EQA-GPT4o' or method == 'Explore-EQA-GPT4o_steps':
        results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gpt4o_early_term/'
        good_pkls = [results_path+"results_130.pkl", results_path+"results_140.pkl", results_path+"results_170.pkl",results_path+"results_190.pkl", results_path+"results_200.pkl", results_path+"results_210.pkl", results_path+"results_250.pkl", results_path+"results_300.pkl", results_path+"results_310.pkl",results_path+"results_320.pkl",results_path+"results_340.pkl", results_path+"results_350.pkl", results_path+"results_500.pkl"] 
        return good_pkls
    
    if method == 'Explore-EQA-GPT4o-0shot':
        results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gpt4o_one_step/'
        good_pkls = [results_path + 'results_440.pkl', results_path + 'results_500.pkl']
        return good_pkls

def load_idx_file():
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/results/sem_to_all_indx.json', 'r') as file:
        sem_to_all_indx = json.load(file)
    semantic_all_idxs = [v['all_idx'] for k,v in sem_to_all_indx.items()]
    return sem_to_all_indx, semantic_all_idxs

if __name__ == "__main__":

    method = 'Explore-EQA-GPT4o_steps'
    questions_data = load_eqa_data()
    sem_to_all_indx, semantic_all_idxs = load_idx_file()
    worksheet, method_col = load_spreadsheet(method)
    good_pkls = get_good_pickles(method)

    for file_name in good_pkls:
        with open(file_name, 'rb') as file:
            results = pickle.load(file)

        for i in trange(len(results)):
            result = results[i]
            if result['question_ind'] in semantic_all_idxs:
                log_succ = False
                while not log_succ:
                    try:
                        q_row = worksheet.find(str(result['question_ind']), in_column=1).row
                        num_max_steps = result['num_max_steps']

                        # if result['success_max'] and result[f'step_{num_max_steps}']['smx_vlm_rel'][0]>0.5:
                        #     worksheet.update_cell(q_row, method_col, 1)
                        # else:
                        #     worksheet.update_cell(q_row, method_col, 0)
                        worksheet.update_cell(q_row, method_col, int(num_max_steps))

                        log_succ = True
                    except Exception as e:
                        print(f"An error occurred: {e}. Sleeping for 60")
                        time.sleep(60)
