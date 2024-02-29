import json
import os
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import tool
from langchain.agents import Tool

from tools.turn_lights_on import TurnLightsOnTool
from tools.turn_lights_off import TurnLightsOffTool
from tools.get_sensor_data import GetSensorDataTool

from dotenv import load_dotenv

# https://www.comet.com/site/blog/enhancing-langchain-agents-with-custom-tools/
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py

load_dotenv()
apiKey = os.environ["OPENAI_KEY"]
llm = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-3.5-turbo", temperature=0)

# Ollama
# llm = Ollama(base_url='http://localhost:11434', model='llama2')

class CustomTool:
    def __init__(self):
        prompt = PromptTemplate(input_variables=["query"], template="{query}")

        # when giving tools to LLM, we must pass as list of tools
        tools = [TurnLightsOffTool(), TurnLightsOnTool(), GetSensorDataTool()]

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
