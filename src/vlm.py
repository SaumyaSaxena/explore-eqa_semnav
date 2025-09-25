import time
import logging
import torch
import numpy as np
from prismatic import load

import google.generativeai as genai
from together import Together

import os
import mimetypes, json
from enum import Enum
import base64

def one_hot_encode(options, choice):
    # Initialize a list of zeros with the same length as options
    encoding = [0.] * len(options)
    
    # Find the index of the choice in options and set that position to 1
    if choice in options:
        encoding[options.index(choice)] = 1.
    else:
        raise ValueError(f"Choice '{choice}' not found in options: {options}")
    
    return np.array(encoding)

class VLM:
    def __init__(self, cfg):
        start_time = time.time()
        self.model = load(cfg.model_id, hf_token=cfg.hf_token)
        self.model.to(cfg.device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        losses = self.model.get_loss(
            image,
            prompt_text,
            return_string_probabilities=tokens,
        )[0]
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses


class GeminiVLM:
    def __init__(self, cfg):
        self.use_image = cfg.use_image
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")

    def get_answer(self, image_path, prompt_question, prompt_confidence, vlm_pred_candidates, choices):
        
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(vlm_pred_candidates, choices)}, type=str)
        messages=[
            {"role": "user", "parts": [{"text": f"{prompt_question} {prompt_confidence}"}]},
        ]
        base64_image = self.encode_image(image_path)
        mime_type = mimetypes.guess_type(image_path)[0]
        messages.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        )
        
        answer = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'answer': genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=[member.name for member in Answer_options]
                ),
                'value': genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=[member.value for member in Answer_options]
                ),
                'is_confident': genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description=f"{prompt_confidence}"
                )
            },
            required=['answer', 'value', 'is_confident']
        )

        response = self.gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=answer),
        )

        json_response = response.text
        response_dict = json.loads(json_response)

        smx_vlm_pred = one_hot_encode(vlm_pred_candidates, response_dict['answer'])
        smx_vlm_rel = [1.0, 0.0] if response_dict['is_confident'] else [0.0, 1.0]
        return smx_vlm_pred, smx_vlm_rel


    def get_frontier_and_gsv(self, prompted_img_path, prompt_lsv, prompt_gsv, draw_letters):
        messages=[
            {"role": "user", "parts": [{"text": f"{prompt_lsv} {prompt_gsv}"}]},
        ]
        base64_image = self.encode_image(prompted_img_path)
        mime_type = mimetypes.guess_type(prompted_img_path)[0]
        messages.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": "CURRENT IMAGE: This image represents the current view of the agent."
                    },
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        )
        
        answer = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'answer': genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=[member for member in draw_letters]
                ),
                'explore_anywhere': genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description=f"{prompt_gsv}"
                )
            },
            required=['answer', 'explore_anywhere']
        )

        response = self.gemini_model.generate_content(
            messages,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                response_schema=answer),
        )

        json_response = response.text
        response_dict = json.loads(json_response)

        lsv = one_hot_encode(draw_letters, response_dict['answer'])
        return lsv, float(response_dict['explore_anywhere'])

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class LlamaVLM:
    def __init__(self, cfg):
        self.client = Together()
        if "TOGETHER_API_KEY" not in os.environ:
            print('Together AI token has not been set up yet!')
        self._vlm_type = cfg.name
        self.use_image = cfg.use_image

    def get_instruction_set(self, Answer_options):

      instructions = f'''
        The output should strictly follow the below JSON format.
        Note that you should choose only one of the two types of available answers for each query.
        You are only allowed to provide the answers available in the set of options provided to you in
        the following list: {', '.join(member.name for member in Answer_options)}. 'None' is not an acceptable choice.
        Before you populate any of the sections below, be sure to NOT wrap any particular word or phrase
        in double quotes. This will cause parsing to not work correctly, and this is needed of the 
        JSON string output. 'None' is not an acceptable answer for 'answer' or 'value'.
        You must select an opton available in the provided list.

        {{
          {{"answer": "select one from: {', '.join(member.value for member in Answer_options)}"}},
          {{"value": "select one from: {', '.join(member.name for member in Answer_options)}"}},
          {{"is_confident": "Return the string 'true' or 'false'. This must be a string."}}
        }}
        '''
      
      return instructions
    
    def get_eqa_instructions(self, Answer_options, prompt_gsv):

      instructions = f'''
        The output should strictly follow the below JSON format.
        Note that you should choose only one of the two types of available answers for each query.
        You are only allowed to provide the answers available in the set of options provided to you in
        the following list: {', '.join(member for member in Answer_options)}. 'None' is not an acceptable choice.
        'None' is not an acceptable choice.
        Before you populate any of the sections below, be sure to NOT wrap any particular word or phrase
        in double quotes. This will cause parsing to not work correctly, and this is needed of the 
        JSON string output.

        {{
          {{"answer": "select one from: {', '.join(member for member in Answer_options)}"}},
          {{"explore_anywhere": "Return a string 'True' or 'False'. {prompt_gsv}"}} 
        }}
        '''
      
      return instructions
    
    def get_answer(self, image_path, prompt_question, prompt_confidence, vlm_pred_candidates, choices):
        
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(vlm_pred_candidates, choices)}, type=str)
        output_instructions = self.get_instruction_set(Answer_options)
        
        messages=[
            {"role": "system", "content": f"{prompt_question} {prompt_confidence}"},
            {"role": "user", "content": f"INSTRUCTIONS: {output_instructions}"}
        ]
        base64_image = self.encode_image(image_path)
        messages.append(
            { 
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            })

        import json, re
        success = False
        response_dict = {'value': None}

        while success == False and response_dict['value'] == None:
            try:
                completion = self.client.chat.completions.create(
                    model="meta-llama/" + self._vlm_type,
                    messages=messages
                )
                response = completion.choices[0].message.content
                json_text = re.search(r'\{.*\}', response, re.DOTALL).group(0)
                response_dict = json.loads(json_text)
                success = True
            except Exception as e:
                print(f"Parsing error: {e}! Retrying...")
                time.sleep(1)

        smx_vlm_pred = one_hot_encode(vlm_pred_candidates, response_dict['value'])
        if response_dict['is_confident'].lower() == 'false':
            confident = False
        else:
            confident = True
        smx_vlm_rel = [1.0, 0.0] if confident else [0.0, 1.0]
        return smx_vlm_pred, smx_vlm_rel

    def get_frontier_and_gsv(self, prompted_img_path, prompt_lsv, prompt_gsv, draw_letters):
        messages=[
            {"role": "user", "content": f"{prompt_lsv} {prompt_gsv}"},
            {"role": "user", "content": f"INSTRUCTIONS: {self.get_eqa_instructions(draw_letters, prompt_gsv)}"}
        ]
        base64_image = self.encode_image(prompted_img_path)
        messages.append(
            { 
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            })

        import json, re
        success = False
        while success == False:
            try:
                completion = self.client.chat.completions.create(
                    model="meta-llama/" + self._vlm_type,
                    messages=messages
                )
                response = completion.choices[0].message.content
                json_text = re.search(r'\{.*\}', response, re.DOTALL).group(0)
                response_dict = json.loads(json_text)
                # import ipdb; ipdb.set_trace()
                success = True
            except Exception as e:
                print(f"Parsing error: {e}! Retrying...")
                time.sleep(1)

        #import ipdb; ipdb.set_trace()
        lsv = one_hot_encode(draw_letters, response_dict['answer'])
        if response_dict['explore_anywhere'].lower() == 'false':
            explore_anywhere = False
        else:
            explore_anywhere = True
        return lsv, float(explore_anywhere)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


from openai import OpenAI
from pydantic import BaseModel

class GPT4oVLM:
    def __init__(self, cfg):
        self.use_image = cfg.use_image
        self.client = OpenAI()
        self._vlm_type = "gpt-4o-2024-08-06"

    def get_answer(self, image_path, prompt_question, prompt_confidence, vlm_pred_candidates, choices):
        
        Answer_options = Enum('Answer_options', {token: choice for token, choice in zip(vlm_pred_candidates, choices)}, type=str)

        messages=[
            {"role": "user", "content": f"{prompt_question} {prompt_confidence}"},
        ]
        if self.use_image:
            base64_image = self.encode_image(image_path)
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
        
        class Answer(BaseModel):
            explanation_ans: str
            answer: Answer_options
            is_confident: bool

        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=Answer,
        )
        for _ in range(10):
            plan = completion.choices[0].message
            if not (plan.refusal): # If the model refuses to respond, you will get a refusal message
                break
        ans = plan.parsed.answer.name
        conf = plan.parsed.is_confident

        smx_vlm_pred = one_hot_encode(vlm_pred_candidates, ans)
        smx_vlm_rel = [1.0, 0.0] if conf else [0.0, 1.0]
        return smx_vlm_pred, smx_vlm_rel


    def get_frontier_and_gsv(self, prompted_img_path, prompt_lsv, prompt_gsv, draw_letters):

        Draw_Letter_options = Enum('Draw_Letter_options', {let: let for let in draw_letters}, type=str)

        messages=[
            {"role": "user", "content": f"{prompt_lsv} {prompt_gsv}"},
        ]
        if self.use_image:
            base64_image = self.encode_image(prompted_img_path)
            messages.append(
                { 
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information to answer the question."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
        
        class DrawLetter(BaseModel):
            explanation_letter: str
            draw_letter: Draw_Letter_options
            explore_anywhere: bool

        completion = self.client.beta.chat.completions.parse(
            model=self._vlm_type,
            messages=messages,
            response_format=DrawLetter,
        )
        plan = completion.choices[0].message
        letter = plan.parsed.draw_letter.name
        explore_anywhere = plan.parsed.explore_anywhere

        lsv = one_hot_encode(draw_letters, letter)
        return lsv, float(explore_anywhere)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')