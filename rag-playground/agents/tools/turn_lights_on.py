from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class RequestBody(BaseModel):
    location: str = Field(description='the location of the light')

class TurnLightsOnTool(BaseTool):
    name = "turn_lights_on"
    description = "Use when you need to turns the lights 'ON' for a given location"
    args_schema: Type[BaseModel] = RequestBody
    return_direct = True

    def _run(
        self, 
        location: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print('!!!!ENTERED!!!!')
        resp = {"location": location, "action": False}
        return f"OK. The {location} lights has been turned on"

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")