import json
import os
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain_core.tools import ToolException
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentType, initialize_agent, load_tools
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
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        llm_math = LLMMathChain(llm=llm)

        # initialize the math tool
        math_tool = Tool(
            name="my_calculator",
            func=llm_math.run,
            description="Useful for when you need to answer questions about math.",
            handle_tool_error=True
        )

        llm_tool = Tool(
            name="Language Model",
            func=llm_chain.run,
            description="use this tool for general purpose queries and logic",
            handle_tool_error=True
        )

        turn_lights_on_tool = Tool(
            name='turn_lights_on',
            func=self.turn_lights_on,
            description='Use this tool when you need to turns the lights on for a given lightId',
            handle_tool_error=True
        )

        turn_lights_off_tool = Tool(
            name='turn_lights_off',
            func=self.turn_lights_off,
            description='Use this tool when you need to turns the lights off for a given lightId',
            handle_tool_error=True
        )

        get_sensor_data = Tool(
            name='get_sensor_data',
            func=self.get_sensor_data,
            description='Use this tool to retrieve sensor data for a given sensorId',
            handle_tool_error=True
        )

        # when giving tools to LLM, we must pass as list of tools
        tools = [math_tool, turn_lights_on_tool, turn_lights_off_tool, get_sensor_data]
        llm_with_tools = llm.bind_tools(tools)

        zero_shot_template = """Answer the following questions as best you can. You have access to the following tools:
            
            my_calculator: Useful for when you need to answer questions about math.
            Language Model: use this tool for general purpose queries and logic
            turn_lights_on: Use this tool when you need to turns the lights on for a given lightId
            turn_lights_off: Use this tool when you need to turns the lights off for a given lightId
            get_sensor_data: Use this tool to retrieve sensor data for a given sensorId
            
            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [my_calculator, Language Model, turn_lights_on, turn_lights_off, get_sensor_data]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!
            
            Question: {input}
            Thought:{agent_scratchpad}')
        """

        template = """
        Answer the following questions as best you can. You have access to the following tools:
            
            my_calculator: Useful for when you need to answer questions about math.
            Language Model: use this tool for general purpose queries and logic
            turn_lights_on: Use this tool when you need to turns the lights on for a given lightId
            turn_lights_off: Use this tool when you need to turns the lights off for a given lightId
            
            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [my_calculator, Language Model, turn_lights_on, turn_lights_off]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!
            
            Question: {input}
            Thought:{agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(template=template)

        # self.zero_shot_agent = create_react_agent(llm, tools, prompt)

        # zero_shot_agent_chain = (
        #     {
        #         "input": lambda x: x["input"],
        #         "tool_names": lambda x: ['my_calculator', 'Language Model', 'turn_lights_on', 'turn_lights_off'],
        #         "agent_scratchpad": lambda x: format_to_openai_tool_messages(
        #             x["intermediate_steps"]
        #         ),
        #     }
        #     | prompt
        #     | llm_with_tools
        #     | OpenAIToolsAgentOutputParser()
        # )

        # self.agent_executor = AgentExecutor(agent=zero_shot_agent_chain, tools=tools, verbose=True, handle_parsing_errors=True)

        PREFIX = """You are very helpful assistant that will answer the following questions as best you can. """

        FORMAT_INSTRUCTIONS = """
            You have access to the following tools.
                my_calculator: Useful for when you need to answer questions about math.
                turn_lights_on: Use this tool when you need to turns the lights on for a given lightId
                turn_lights_off: Use this tool when you need to turns the lights off for a given lightId

            You should only use the tools only if the question is specifically about lights and arithmetic.

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [my_calculator, turn_lights_on, turn_lights_off]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer.
            Final Answer: the final answer to the original input question
        
            Question: Why is the sky blue?
            Thought: This question cannot be appropriately answered by any of the Tools.
            Observable: The question cannot be answered.
            Final Answer: Sorry, I cannot answer that question.
        """

        SUFFIX = """Begin!
            Question: {input}
            Thought:{agent_scratchpad}
        """

        self.zero_shot_agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            return_intermediate_steps=True,
            agent_kwargs={
                'prefix':PREFIX,
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )

        # print(self.zero_shot_agent)

        # zero_shot_agent("what is (4.5*2.1)^2.2?")
        # zero_shot_agent("Good morning")
        # zero_shot_agent("What is the capital of Norway?")

    def _handle_error(self, error) -> str:
        print('FUBAR')
        return str(error) 

    @tool
    def turn_lights_on(lightId: str) -> str:
        """Use this tool when you need to turns the lights on for a given lightId"""
        resp = {
            "lightId": lightId,
            "action": True
        }
        # return json.dumps(resp)
        return f'DONE. The {lightId} lights has been FUBAR on'

    @tool
    def turn_lights_off(lightId: str) -> str:
        """Use this tool when you need to turns the lights off for a given lightId"""
        resp = {
            "lightId": lightId,
            "action": False
        }
        return f'DONE. The {lightId} lights has been FUBAR off'
        # return json.dumps(resp)

    @tool
    def get_sensor_data(sensorId: str) -> str:
        """Use this tool to retrieve sensor data for a given sensorId"""
        sensor_data = {'temperature': 11.1, 'humidity': 45.2}
        resp = {
            "sensorId": sensorId,
            "data": json.dumps(sensor_data)
        }

        return f'DONE. Here is the sensor data.  {json.dumps(resp)}'

    def run(self):
        while True:
            user_input = input(">> ")
            if user_input.lower() == "bye":
                print("LLM: Goodbye")
                break

            if user_input is not None:
                response = self.zero_shot_agent.invoke({"input": user_input})
                # response = self.agent_executor.invoke({"input": user_input})
                # print(response['output'])
                print(response)

        # template = """
        # You are a friendly Shaolin Master that is ready to guide and share wisdom.
        # """
        # prompt = ChatPromptTemplate.from_template(template=template)
        # chain = prompt | llm | StrOutputParser()

        # response = chain.invoke({"input": "hello? My name is Rachel."})
        # print(response)


if __name__ == "__main__":
    customTool = CustomTool()
    customTool.run()