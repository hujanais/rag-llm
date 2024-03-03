import json
from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class RequestBody(BaseModel):
    location: str = Field(description='the location of the sensor')

class GetSensorDataTool(BaseTool):
    name = "get_sensor_data"
    description = "Use this tool to retrieve sensor data in the requested location"
    args_schema: Type[BaseModel] = RequestBody

    def _run(
        self, 
        location: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print('!!!!ENTERED!!!!')
        sensor_data = {"temperature": 11.1, "humidity": 45.2}
        resp = {"location": location, "data": json.dumps(sensor_data)}

        return f"DONE. Here is the sensor data.  {json.dumps(resp)}"


    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")