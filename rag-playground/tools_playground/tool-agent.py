import json
import os
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description_and_args
from tools.tools import CustomSearchTool, CustomCalculatorTool

from dotenv import load_dotenv

from langchain.globals import set_debug
set_debug(True)

# https://www.comet.com/site/blog/enhancing-langchain-agents-with-custom-tools/
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py

load_dotenv()
apiKey = os.environ["OPENAI_KEY"]
llm = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-3.5-turbo", temperature=0)

# Ollama
# llm = Ollama(base_url='http://localhost:11434', model='mistral')


class CustomTool:
    def __init__(self):
        prompt = PromptTemplate(input_variables=["query"], template="{query}")

        llm_chain_tool = Tool(
            name="llm_chain_tool",
            func=self.llm_chain_tool,
            description="Use this tool for general queries",
            handle_tool_error=True,
        )

        # when giving tools to LLM, we must pass as list of tools
        tools = [CustomSearchTool(), CustomCalculatorTool(), llm_chain_tool]

        template = """
            You are very helpful assistant that will only answer using custom tools and never from memory.

            You have access to the following tools.
            {tool_list}

            Use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer.
            Final Answer: the final answer to the original input question
        
            Begin!
            
            Question: {input}
            Thought: {agent_scratchpad}
        """
        prompt = ChatPromptTemplate.from_template(template=template)
        prompt = prompt.partial(
            tool_list=render_text_description(list(tools)),
            tool_names=", ".join([t.name for t in tools]),
        )
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

    @tool
    def llm_chain_tool(lightId: str) -> str:
        """Use this tool for that requires general knowledge"""
        return 'Sorry, I am unable to answer this question as it is out of my domain knowledge.'

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
