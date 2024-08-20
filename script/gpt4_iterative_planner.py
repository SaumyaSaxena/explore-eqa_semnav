import json
from enum import Enum
from typing import Union

from openai import OpenAI
import tiktoken
from src.scene_graph_sim import SceneGraphSim

client = OpenAI()

from pydantic import BaseModel

class Manipulation_action(str, Enum):
    Access = "access"
    Pickup = "pickup"
    Release = "release"
    Open = "open"
    Close = "close"
    TurnOn = "turnon"
    TurnOff = "turnoff"
    Done = "done_with_task"

class Manipulation_action_effect(str, Enum):
    Access = "accessed"
    Pickup = "picked up"
    Release = "released"
    Open = "opened"
    Close = "closed"
    TurnOn = "turned on"
    TurnOff = "turned off"
    Done = "done_with_task"

class Navigation_action(str, Enum):
    Goto_new_room = "goto_new_room"
    Goto_visited_room = "goto_visited_room"

class Navigation_action_effect(str, Enum):
    Goto_new_room = "In new room"
    Goto_visited_room = "In room"

class Manipulation_step(BaseModel):
    explanation_env: str
    action: Manipulation_action
    object_name: str

class Navigation_step(BaseModel):
    explanation_exp: str
    action: Navigation_action
    room_name: str

class PlannerResponse(BaseModel):
    steps: list[Union[Navigation_step, Manipulation_step]]
    final_full_plan: str

def json_to_prompt():
    with open("/home/saumyas/Projects/semnav/3DSceneGraph/3dsg.json", "r") as f:
        sg = json.load(f)
    json_string = json.dumps(sg)
    return json_string

def agent_role_prompt():
    prompt = "You are an excellent graph planning agent. \
        You are given a scene graph representation (in json format) of the areas of the environment you have explored so far. \
        You navigate your environment using 'Navigation_step'. 'Goto_visited_room' will take you to an already visited room specified by 'room_name'. 'Goto_new_room' action will take you to the closest unexplored room and expand the scene graph. Use 'room_name'=''(empty) for this action. \n \
        You can manipulate the visible environment (specified in the scene graph) using 'Manipulation_action' and 'object_name'. \n \
        Current state of the agent is represented as a sequence of {'State', 'Action', 'Effect'} triplets which represent the history of states, actions and effects so far. \n \
        When you are confident that the task is complete, take action: done_with_task. \n \
        Since you will be planning for long-horizon tasks, you need to consider the current scene graph as well the history of steps taken so far, to generate a step-by-step task plan that \
        you should follow to solve a given instruction."
    return prompt

def agent_state_prompt(room):
    prompt = f'You are currently in room: {room}' # TODO: should this be more than room info
    return prompt

def memory_prompt_str(memory, agent_state, step):
    new_memory = f'{memory}  Next Step:  Visited state: "{agent_state}" and took action: "{step}".'
    return new_memory

def memory_prompt_json(memory, curr_state, step=None, new_state=None, t=0):
    if isinstance(step, Navigation_step):
        action_str = f'{step.action.value}<{step.room_name}>'
        effect_str = f'{Navigation_action_effect[step.action.name].value}: {new_state}'
    elif isinstance(step, Manipulation_step):
        action_str = f'{step.action.value}<{step.object_name}>'
        effect_str = f'{Manipulation_action_effect[step.action.name].value}: {step.object_name}'
    elif step is None:
        action_str = ''
        effect_str = ''
    else:
        raise NotImplementedError('Not implemented memory')

    new_memory = {
        f'State_t={t}': curr_state,
        f'Action_t={t}': action_str,
        f'Effect_t={t}': effect_str
    }
    memory.append(new_memory)
    return memory

def instruction():
    # prompt = 'Get an apple from the refrigerator and take it to the bathroom.'
    prompt = 'Goto refrigerator and then goto the bathroom.'
    # prompt = 'Get an apple from the refrigerator.'
    # prompt = 'Goto kitchen and then goto bathroom.'
    return prompt

def append_to_json_file(new_messages, filename):
    try:
        # Step 1: Read the existing JSON data from the file
        with open(filename, 'r') as file:
            messages = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        messages = []

    # Step 2: Append new messages to the existing list
    if isinstance(messages, list):  # Ensure the content is a list
        messages.extend(new_messages)
    else:
        raise ValueError("The existing content is not a list. Cannot append.")

    # Step 3: Write the updated list back to the JSON file
    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)

def count_tokens(inp):
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(inp)
    return len(tokens)

if __name__== "__main__":

    output_file = "plan.json"
    agent_role = agent_role_prompt()
    instr = instruction()

    scene_graph_sim = SceneGraphSim()
    scene_graph = scene_graph_sim.start()

    curr_state = scene_graph_sim.get_current_state() # TODO: dhould this come from scene_graph_sim?
    curr_state_prompt = agent_state_prompt(curr_state)
    
    
    memory = memory_prompt_json([], curr_state)
    # print("Scene graph tokens:", count_tokens(scene_graph))    
    done = False
    t = 0
    append_to_json_file([{"Instruction": instr}], output_file)
    while not done:
        messages=[
            {"role": "system", "content": f"{agent_role}"},
            {"role": "user", "content": f"Current state (+full history of observations): {json.dumps(memory)}. \
                Current room: {curr_state_prompt}. \
                Instruction: {instr}. \
                Scene Graph in json format: {scene_graph}."},
        ]
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            # model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=PlannerResponse,
        )

        plan = completion.choices[0].message
        # If the model refuses to respond, you will get a refusal message
        if (plan.refusal):
            print(plan.refusal)
        else:
            t += 1
            step_disc = " ".join([
                v.value if isinstance(v, Enum) else v 
                for v in list(plan.parsed.steps[0].model_dump().values()) 
            ][1:])
            
            print(f'[State: {curr_state}, Action: {step_disc}]')
            outputs = [{"Memory": json.dumps(memory), "Current state": curr_state, 'Action': step_disc, "Desc": list(plan.parsed.steps[0].model_dump().values())[0]}]
            print(plan.parsed)
            
            if 'done_with_task' in plan.parsed.steps[0].action.value:
                done = True
                outputs.append({"Done": 'done_with_task'})
                print("DONE WITH TASK")
            else:
                # Execute first step of the plan
                if 'goto_next_room' in plan.parsed.steps[0].action.value:
                    # Update the scene graph
                    scene_graph, sg_done = scene_graph_sim.explore_next_room()
                    if sg_done:
                        done = True
                        outputs.append({"Done": 'scene_fully_explored'})
                        print("SCENE FULLY EXPLORED") # TODO: reexplore explored regions
                    new_state = scene_graph_sim.get_current_state() #TODO: for now only changes when room is changed
                    curr_state_prompt = agent_state_prompt(new_state)
                    
                    memory = memory_prompt_json(memory, curr_state, plan.parsed.steps[0], new_state, t=t)
                    curr_state = new_state

        append_to_json_file(outputs, output_file)
        
    