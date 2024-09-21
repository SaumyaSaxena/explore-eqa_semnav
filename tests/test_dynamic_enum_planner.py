import json
from enum import Enum

from openai import OpenAI
import tiktoken

client = OpenAI()

from pydantic import BaseModel


def create_planner_response(Action):
    class Step(BaseModel):
        explanation: str
        step: Action

    class PlannerResponse(BaseModel):
        steps: list[Step]
        final_full_plan: str
    return PlannerResponse


if __name__== "__main__":

    actions = ['goto', 'pickup', 'throw']
    Action = Enum('Action', {ac: ac for ac in actions}, type=str)

    for i in range(2):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            # model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an agent taking actions at random"},
                {"role": "user", "content": f"Instruction: Choose a random action."},
            ],
            response_format=create_planner_response(Action),
        )

        actions = ['open', 'close', 'turn_on', 'turn_off']
        Action = Enum('Action', {ac: ac for ac in actions}, type=str)

        plan = completion.choices[0].message
        if (plan.refusal):
            print(plan.refusal)
        else:
            print(plan.parsed)
    