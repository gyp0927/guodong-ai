from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """多Agent共享状态"""
    messages: Annotated[Sequence[BaseMessage], add]
    active_agent: str | None
    task_context: dict | None
    human_input_required: bool
    base_model_response: str | None
    review_result: str | None
    awaiting_review: bool