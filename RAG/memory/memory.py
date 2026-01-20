# memory/memory.py
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any


class ConversationBufferMemory:
    """Simple conversation buffer memory implementation"""

    def __init__(self, memory_key: str = "chat_history", return_messages: bool = True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory: List[BaseMessage] = []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation turn"""
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")

        self.chat_memory.append(HumanMessage(content=input_str))
        self.chat_memory.append(AIMessage(content=output_str))

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer"""
        return {
            self.memory_key: (
                self.chat_memory
                if self.return_messages
                else self._get_messages_as_string()
            )
        }

    def _get_messages_as_string(self) -> str:
        """Convert messages to string format"""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in self.chat_memory])

    def clear(self) -> None:
        """Clear memory contents"""
        self.chat_memory = []


def get_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
