import json
from src.utils import load_openeqa_data
from omegaconf import OmegaConf
import numpy as np

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()
class Choices(BaseModel):
    choice1: str
    choice2: str
    choice3: str

def get_choices_answer_id(question, answer):

    prompt = f'''
        Your task is to create a multiple choice question. 
        You are given the question and one of the possible answers. 
        Generate three more possible answers. Keep them short.
        Question: {question}.
        Answer: {answer}.
    '''
    
    messages=[
        {"role": "user", "content": f"{prompt}"},
    ]
    
    vlm_type = "gpt-4o-2024-08-06"
    completion = client.beta.chat.completions.parse(
        model=vlm_type,
        messages=messages,
        response_format=Choices,
    )

    plan = completion.choices[0].message
    for _ in range(10):
        plan = completion.choices[0].message
        if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
            break
    choices = [plan.parsed.choice1, plan.parsed.choice2, plan.parsed.choice3]
    return choices

if __name__ == "__main__":
    choices_save_pth = '/home/saumyas/Projects/semnav/explore-eqa_semnav/data/open-eqa-choices.json'
    cfg_file = '/home/saumyas/Projects/semnav/explore-eqa_semnav/cfg/openeqa_exp.yaml'
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)

    questions_data, init_pose_data = load_openeqa_data(cfg)

    choices_dict = {}
    # Create choices for each question
    for i, data in enumerate(questions_data):
        question = data['question']
        choices = get_choices_answer_id(data['question'], data['answer'])
        # answer_id = np.random.choice(['A', 'B', 'C', 'D'])

        answer_letters = ['A', 'B', 'C', 'D']
        answer_id = np.random.choice(np.arange(4))
        answer_letter = answer_letters[answer_id]
        choices.insert(answer_id, data['answer'])

        choices_dict[data['question_id']] = {
            'question': data['question'],
            'choices': choices,
            'answer': data['answer'],
            'answer_id': answer_letter,
        }

        print(f' {i}: Episode Question: {data["question"]}. \n Answer: {data["answer"]}. \n  Choices: {choices}. \n Answer Letter: {answer_letter}')

        with open(choices_save_pth, 'w') as json_file:
            json.dump(choices_dict, json_file, indent=4)