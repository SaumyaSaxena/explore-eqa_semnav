import json
from enum import Enum
from typing import Union

from openai import OpenAI
import tiktoken
from src.scene_graph_sim import SceneGraphSim
from pydantic import BaseModel

class Manipulation_action(str, Enum):
    Access = "access"
    Pickup = "pickup"
    Release = "release"
    Open = "open"
    Close = "close"
    TurnOn = "turnon"
    TurnOff = "turnoff"

class Manipulation_action_effect(str, Enum):
    Access = "accessed"
    Pickup = "picked up"
    Release = "released"
    Open = "opened"
    Close = "closed"
    TurnOn = "turned on"
    TurnOff = "turned off"

class Navigation_action(str, Enum):
    Goto_visited_room = "goto_visited_room"

class Navigation_action_effect(str, Enum):
    Goto_visited_room = "In room"

class Navigation_Exploration_action(str, Enum):
    Goto_new_room = "goto_new_room"
    # Done_exploring = "done_exploring"

class Navigation_Exploration_action_effect(str, Enum):
    Goto_new_room = "In new room"
    # Done_exploring = "done_exploring"

class Done_action(str, Enum):
    Done = "done_with_task"

class Done_action_effect(str, Enum):
    Done = "done_with_task"

class Manipulation_step(BaseModel):
    explanation_env: str
    action: Manipulation_action
    object_name: str

class Navigation_step(BaseModel):
    explanation_nav: str
    action: Navigation_action
    room_id: str

class Navigation_Exploration_step(BaseModel):
    explanation_exp: str
    action: Navigation_Exploration_action

class Done_step(BaseModel):
    explanation_done: str
    action: Done_action

class PlannerResponse(BaseModel):
    steps: list[Union[Navigation_Exploration_step, Navigation_step, Manipulation_step, Done_step]]
    final_full_plan: str


def agent_role_prompt():
    prompt = "You are an excellent graph planning agent. \
        You are given a scene graph representation (in json format) of the areas of the environment you have explored so far. \
        The scene graph will give you information about the 'rooms' and 'objects' in the scene.\
        Every scene graph node with 'type': 'Room' is a room you can navigate to and node with 'type': 'Object' is object you can manipulate. \n \
        You can take four kinds of steps in the environement: Exploration_step, Navigation_step, Manipulation_step. \n\
        1) Navigation_Exploration_step: 'goto_new_room' action allows you to navigate to a new unexplored room and augment the scene graph with the new observations. If the current state tells you that you are in right room to perform a manipulation_step and do not need to goto a new room, take action: 'done_exploring'. \n \
        2) Navigation_step: 'Goto_visited_room' action will take you to an already visited room specified by 'room_id' in the scene graph. This will not augment the scene graph and not give you any new observations. \n \
        3) Manipulation_step:  'access', 'pickup', 'release', 'open', 'close', 'turnon', and 'turnoff' actions allow you to manipulate objects specified by 'object_name' in the scene graph. \n \
        4) Done_step: Take this step, if you are confident that all necessary steps have been taken and the task is complete. \n \
        Current state of the agent is represented as a sequence of {'State', 'Action', 'Effect'} triplets which represent the history of states, actions and effects so far. \n \
        Since you will be planning for long-horizon tasks, you need to consider the current scene graph as well the history of steps taken so far, to generate a step-by-step task plan that \
        you should follow to solve a given instruction."
    return prompt

def agent_state_prompt(room):
    prompt = f' You are in room: {room}' # TODO: should this be more than room info
    return prompt


def action_effect_str(step, curr_state, new_state):
    if isinstance(step, Navigation_step):
        action_str = f'{step.action.value}<{new_state}>'
        effect_str = f'{Navigation_action_effect[step.action.name].value} <{new_state}> and scene graph not expanded'
    elif isinstance(step, Navigation_Exploration_step):
        action_str = f'{step.action.value}'
        effect_str = f'{Navigation_Exploration_action_effect[step.action.name].value} <{new_state}> and expanded scene graph'
    elif isinstance(step, Manipulation_step):
        action_str = f'{step.action.value}<{step.object_name}>'
        effect_str = f'In room <{curr_state}> and {Manipulation_action_effect[step.action.name].value} <{step.object_name}>'
    elif isinstance(step, Done_step):
        action_str = f'{step.action.value}'
        effect_str = f'{Done_action_effect[step.action.name].value}'
    elif step is None:
        action_str = ''
        effect_str = ''
    else:
        raise NotImplementedError('Not implemented memory')

    return action_str, effect_str

def update_history(history, curr_state, action_str, effect_str, t):
    history_new = f'  [State(t={t}): {curr_state}, Action(t={t}): {action_str}, Effect(t={t}): {effect_str}],  '
    return history + history_new

def step_prompt(curr_state_prompt, history, scene_graph, action_str, effect_str, t):
    prompt = f"At t = {t}: \n \
        USER INPUTS: {curr_state_prompt}. \n \
        History: {history}. \n \
        Scene graph: {scene_graph}. \n \
        AGENT OUTPUT: Action by agent: {action_str}. \n \
        Effect: {effect_str}. \n \n"
    return prompt

def instruction():
    prompt = 'Turn on the microwave.'
    return prompt

if __name__== "__main__":
    
    output_file = "outputs/in_context_example.txt"
    scene_graph_sim = SceneGraphSim(full_sg_path='/home/saumyas/Projects/semnav/3DSceneGraph/3dsg_small.json')

    agent_role = agent_role_prompt()
    instr = instruction()
    role_prompt = f"Agent role: {agent_role} \n \
        Instruction: {instr}"
    
    t, done, step_prompts = 0, False, []
    scene_graph = scene_graph_sim.start(init_room='room_24') # lobby
    scene_graph_new = scene_graph
    curr_state = scene_graph_sim.get_current_state()
    curr_state_prompt = agent_state_prompt(curr_state)
    history = ''
    
    # Plan steps after reaching kitchen
    plan_idx = 0
    manip_actions = [
        Manipulation_step(
            explanation_env='',
            action=Manipulation_action.Access,
            object_name='microwave'),
        Manipulation_step(
            explanation_env='',
            action=Manipulation_action.TurnOn,
            object_name='microwave'),
        Done_step(
            explanation_done='',
            action=Done_action.Done,
        )
    ]
    while not done:
        
        # Plan
        if 'kitchen' not in curr_state.lower():
            step = Navigation_Exploration_step(
                explanation_exp='',
                action=Navigation_Exploration_action.Goto_new_room)
        else:
            step = manip_actions[plan_idx]
            plan_idx+=1

        
        if 'done_with_task' in step.action.value:
            done = True
        else:
            # Execute first step of the plan
            if 'goto_new_room' in step.action.value:
                # Update the scene graph
                scene_graph_new, sg_done = scene_graph_sim.explore_next_room()
                if sg_done:
                    done = True
                
            if "goto_visited_room" in step.action.value:
                scene_graph_sim.goto_room(step.room_id)
                
        
        new_state = scene_graph_sim.get_current_state() #TODO: for now only changes when room is changed

        action_str, effect_str = action_effect_str(step, curr_state, new_state)

        step_prompts.append(step_prompt(curr_state_prompt, history, scene_graph, action_str, effect_str, t))
        
        # Update
        scene_graph = scene_graph_new
        history = update_history(history, curr_state, action_str, effect_str, t)

        curr_state = new_state
        curr_state_prompt = agent_state_prompt(curr_state)
        
        t += 1
    
    full_example = ' '.join(step_prompts)
    with open(output_file, "w") as text_file:
        text_file.write(full_example)