from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class RequestBody(BaseModel):
    lightId: str = Field(description='the id of the lightId')

class TurnLightsOnTool(BaseTool):
    name = "turn_lights_on"
    description = "Use when you need to turns the lights on for a given lightId"
    args_schema: Type[BaseModel] = RequestBody

    def _run(
        self, 
        lightId: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        resp = {"lightId": lightId, "action": False}
        return f"DONE. The {lightId} lights has been FUBAR on"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")