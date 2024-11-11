import os
import enum, json, typing
from pydantic import BaseModel
from typing import Annotated

import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class Answer(enum.Enum):
    YES = "Yes"
    NO = "No"

class InstrumentClass(enum.Enum):
    PERCUSSION = "Percussion"
    STRING = "String"
    WOODWIND = "Woodwind"
    BRASS = "Brass"
    KEYBOARD = "Keyboard"

class SportClass(enum.Enum):
    HOCKEY = "Hockey"
    CRICKET = "Cricket"
    FOOTBALL = "Football"
    TENNIS = "Tennis"

class InstrumentClassStep(typing.TypedDict):
    explanation_inst: Annotated[str, "Explain reasoning."]
    instr: InstrumentClass

class SportClassStep(typing.TypedDict):
    explanation_frontier: Annotated[str, "Explain reasoning."]
    spor: SportClass

class take_step(enum.Enum):
    instrument = InstrumentClass
    sport = SportClass

if __name__== "__main__":
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    messages=[
        {"role": "model", "parts": [{"text": "what category is a flute"}]},
    ]

    step_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        enum=take_step
    )
    steps = genai.protos.Schema(
        type = genai.protos.Type.ARRAY,
        items = step_schema,
        min_items = 1
    )
    answer = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        enum=Answer
    )
    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties = {
            'steps': steps,
            # 'answer': answer,
        },
        required=['steps']
    )

    response = gemini_model.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", 
            response_schema=response_schema,
            ),
    )
    json_response = response.text
    response_dict = json.loads(json_response)
    # _step = response_dict["steps"][0]
    # confidence = response_dict["answer"]

    import ipdb; ipdb.set_trace()