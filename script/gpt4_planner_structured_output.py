import json
from enum import Enum

from openai import OpenAI
import tiktoken

client = OpenAI()

from pydantic import BaseModel

class Action(str, Enum):
    Goto = "goto"
    Pickup = "pickup"
    Release = "release"
    Open = "open"
    Close = "close"
    TurnOn = "turnon"
    TurnOff = "turnoff"
    Done = "done"

class Step(BaseModel):
    explanation: str
    step: Action
    object: str

class PlannerResponse(BaseModel):
    steps: list[Step]
    final_full_plan: str

def json_to_prompt():
    with open("/home/saumyas/Projects/semnav/3DSceneGraph/3dsg.json", "r") as f:
        sg = json.load(f)
    json_string = json.dumps(sg)
    return json_string

def agent_role_prompt():
    prompt = "You are an excellent graph planning agent. \
        You are given a graph representation of an environment. \
        You can use this graph to generate a step-by-step task plan that \
        the agent can follow to solve a given instruction."
    return prompt

def output_response_format():
    prompt = "Output Response Format: \
        Chain of thought: Break your problem down into a series of \
            intermediate reasoning steps to help you determine your next command. \
        Reasoning: Justify why the next action is important. \
        Plan: Task plan consisting of commands: Goto(<node_name>) where <node_name> \
            is the name of the object to perform an operation on."
    return prompt

def instruction():
    prompt = 'You are currently in the lobby. Get an apple from the refrigerator to the bathroom.'
    return prompt

def example():
    prompt = " Example 1: \
            Instruction: Find me something to eat out of. Output:\
            Chain of thought: I have found a bowl on the graph -> The plate has affordance to eat out of -> \
                I will find a plan to goto the plate. \
            Reasoning: I will generate a task plan using the identified object. \
            Environment actions: Goto(<object/room>),  Pick(<object>), Open(<object>). \
            Plan: Goto(bedroom) -> Pick(FloorA) -> Goto(kitchen) -> goto(table) -> goto(bowl) \
            Example 2: \
            Instruction: Move object from bedroom to kitchen. Output:\
            Chain of thought: I have found a bowl on the graph -> The plate has affordance to eat out of -> \
                I will find a plan to goto the plate. \
            Reasoning: I will generate a task plan using the identified object. \
            Plan: Goto(bedroom) -> Pick(FloorA) -> Goto(kitchen) -> goto(table) -> goto(bowl)       "
    return prompt

def count_tokens(inp):
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(inp)
    return len(tokens)

if __name__== "__main__":
    agent_role = agent_role_prompt()
    scene_graph = json_to_prompt()
    example_plan = example()
    instr = instruction()
    output_format = output_response_format()

    # print("Scene graph tokens:", count_tokens(scene_graph))

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{agent_role}"},
            {"role": "user", "content": f"Instruction: {instr}. Graph in json format: {scene_graph}."},
            # {"role": "assistant", "content": f"{example_plan}"},
        ],
        response_format=PlannerResponse,
    )

    plan = completion.choices[0].message
    # If the model refuses to respond, you will get a refusal message
    if (plan.refusal):
        print(plan.refusal)
    else:
        print(plan.parsed)
    