from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class ChatState(TypedDict): 
    messages: Annotated[list, add_messages]