from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class RequestBody(BaseModel):
    query: str = Field(description='query')

class CatchAllChainTool(BaseTool):
    name = "llm_tool"
    description = "Use when you have any queries that does not involve lights, temperature, humidity or sensors."
    args_schema: Type[BaseModel] = RequestBody
    return_direct = True

    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print('!!!!ENTERED!!!!')
        return 'Sorry, I am unable to get your answer.'

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")