import json
import os
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import tool
from langchain.agents import Tool

from dotenv import load_dotenv

# https://www.comet.com/site/blog/enhancing-langchain-agents-with-custom-tools/

load_dotenv()
apiKey = os.environ["OPENAI_KEY"]
llm = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-3.5-turbo", temperature=0)


class CustomTool:
    def __init__(self):
        prompt = PromptTemplate(input_variables=["query"], template="{query}")

        turn_lights_on_tool = Tool(
            name="turn_lights_on",
            func=self.turn_lights_on,
            description="Use this tool when you need to turns the lights on for a given lightId",
            handle_tool_error=True,
        )

        turn_lights_off_tool = Tool(
            name="turn_lights_off",
            func=self.turn_lights_off,
            description="Use this tool when you need to turns the lights off for a given lightId",
            handle_tool_error=True,
        )

        get_sensor_data = Tool(
            name="get_sensor_data",
            func=self.get_sensor_data,
            description="Use this tool to retrieve sensor data for a given sensorId",
            handle_tool_error=True,
        )

        # when giving tools to LLM, we must pass as list of tools
        tools = [turn_lights_on_tool, turn_lights_off_tool, get_sensor_data]

        template = """
            You are very helpful assistant that will answer the following questions as best you can. 
            
            You have access to the following tools.
            {tools}

            You should only use the tools only if the question is specifically about lights, sensors and arithmetic.

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer.
            Final Answer: the final answer to the original input question
        
            Question: Why is the sky blue?
            Thought: This question cannot be appropriately answered by any of the Tools.
            Observable: The question cannot be answered.
            Final Answer: Sorry, I cannot answer that question.

            Begin!
            
            Question: {input}
            Thought:{agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(template=template)
        prompt = prompt.partial(
            tools=render_text_description(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )

        pprint(prompt.input_variables)
        pprint(prompt.partial_variables)

        llm_with_stop = llm.bind(stop=["\nObservation"])
        zero_shot_agent_chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
        )

        self.agent_executor = AgentExecutor(agent=zero_shot_agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)

    def _handle_error(self, error) -> str:
        print("FUBAR")
        return str(error)

    @tool
    def turn_lights_on(lightId: str) -> str:
        """Use this tool when you need to turns the lights on for a given lightId"""
        resp = {"lightId": lightId, "action": True}
        # return json.dumps(resp)
        return f"DONE. The {lightId} lights has been FUBAR on"

    @tool
    def turn_lights_off(lightId: str) -> str:
        """Use this tool when you need to turns the lights off for a given lightId"""
        resp = {"lightId": lightId, "action": False}
        return f"DONE. The {lightId} lights has been FUBAR off"
        # return json.dumps(resp)

    @tool
    def get_sensor_data(sensorId: str) -> str:
        """Use this tool to retrieve sensor data for a given sensorId"""
        sensor_data = {"temperature": 11.1, "humidity": 45.2}
        resp = {"sensorId": sensorId, "data": json.dumps(sensor_data)}

        return f"DONE. Here is the sensor data.  {json.dumps(resp)}"

    def run(self):
        while True:
            user_input = input(">> ")
            if user_input.lower() == "bye":
                print("LLM: Goodbye")
                break

            if user_input is not None:
                # response = self.zero_shot_agent.invoke({"input": user_input})
                response = self.agent_executor.invoke({"input": user_input})
                # print(response['output'])
                print(response)

if __name__ == "__main__":
    customTool = CustomTool()
    customTool.run()
