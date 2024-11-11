import time
import logging
import torch
import numpy as np
from prismatic import load

import google.generativeai as genai
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
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

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
            response_schema=answer),
        )

        json_response = response.text
        response_dict = json.loads(json_response)

        lsv = one_hot_encode(draw_letters, response_dict['answer'])
        return lsv, float(response_dict['explore_anywhere'])
    
    def get_global_exploration_value(self, image, prompt):
        pass

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

        messages=[
            {"role": "user", "parts": [{"text": f"{prompt}"}]},
        ]
        self.gemini_model.generate_content("Explain how AI works")

        return generated_text

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
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