import gspread
import pickle, time
from tqdm import trange
# https://docs.gspread.org/en/latest/user-guide.html

if __name__ == "__main__":
    gc = gspread.service_account('/home/saumyas/.config/gspread/saumya-default-project-8b699ae84be9.json')
    sh = gc.open("GraphEQA results 2024")
    # worksheet = sh.worksheet("Analysis zeroshot")
    worksheet = sh.worksheet("Sheet3")
    # print(worksheet.acell('A1').value)

    method_col = worksheet.find('Explore-EQA-GPT4o', in_row=1).col
    
    results_path = '/home/saumyas/Projects/semnav/explore-eqa_semnav/results/gpt4o_one_step/'
    with open(results_path + 'results_500.pkl', 'rb') as file:
        results = pickle.load(file)

    for i in trange(len(results)):
        result = results[i]
        
        log_succ = False
        while not log_succ:
            try:
                q_row = worksheet.find(str(result['question_ind']), in_column=1).row
                if result['success_max'] and result['step_0']['smx_vlm_rel'][0]>0.5:
                    worksheet.update_cell(q_row, method_col, 1)
                else:
                    worksheet.update_cell(q_row, method_col, 0)
                log_succ = True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 60")
                time.sleep(60)