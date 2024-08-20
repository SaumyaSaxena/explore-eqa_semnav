import json

from openai import OpenAI
import tiktoken

client = OpenAI()

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
    # prompt = 'Find me something to eat out of'
    prompt = 'Pick up bowl from bedroom and take to living room?'
    # prompt = 'Find me a bowl'
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

    print("Scene graph tokens:", count_tokens(scene_graph))

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{agent_role}"},
            {"role": "user", "content": f"Instruction: {instr}. Graph in json format: {scene_graph}. Output format: {output_format}"},
            {"role": "assistant", "content": f"{example_plan}"},
        ],
    )

    print(completion.choices[0].message)
    