from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Dict, List, Any


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM was: *** \n{prompts[0]}")
        print("********")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """RUN when LLM ends running."""
        print(f"**** LLM Response: ***\n{response.generations[0][0].text}")
        print("********")
