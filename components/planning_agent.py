from typing import List, Annotated
from fastapi import Body
from eidolon_ai_sdk.apu.apu import APU, Thread
from eidolon_ai_sdk.apu.agent_io import UserTextAPUMessage, SystemAPUMessage
from eidolon_ai_client.events import AgentStateEvent
from eidolon_ai_sdk.system.agent_builder import AgentBuilder


class PlanningAgent(AgentBuilder):
    planning_system_prompt: str = (
        "You are a helpful assistant. Users will come to you with questions. "
        "You will respond with a list of steps to follow when answering the question. "
        "Consider available tools when creating a plan, but do not execute them. "
        "Think carefully."
    )
    agent_system_prompt: str = "You are a helpful assistant"
    user_prompt_template: str = "{user_message}\n\nFollow the execution plan below:\n{steps}"


@PlanningAgent.action(allowed_states=["initialized", "idle"])
async def converse(process_id: str, user_message: Annotated[str, Body()], spec: PlanningAgent):
  """Respond to user messages after creating an execution plan."""
  # Create a response plan
  apu: APU = spec.apu_instance()
  planning_thread: Thread = apu.new_thread(process_id)
  steps: List[str] = await planning_thread.run_request(
    prompts=[
      SystemAPUMessage(prompt=spec.planning_system_prompt),
      UserTextAPUMessage(prompt=user_message)
    ],
    output_format=List[str]
  )

  # Execute the plan
  steps_formatted = "\n".join([f"<step>{step}</step>" for step in steps])
  user_prompt = spec.user_prompt_template.format(user_message=user_message, steps=steps_formatted)
  async for event in apu.main_thread(process_id).stream_request(
    boot_messages=[SystemAPUMessage(prompt=spec.agent_system_prompt)],
    prompts=[UserTextAPUMessage(prompt=user_prompt)]
  ):
    yield event
  yield AgentStateEvent(state="idle")
