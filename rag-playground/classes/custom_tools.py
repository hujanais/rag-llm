import os
import pprint
import random
from json import tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain.agents import Tool

from dotenv import load_dotenv

load_dotenv()
apiKey = os.environ["OPENAI_KEY"]
llm = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-3.5-turbo", temperature=0)


class CustomTool:
    def __init__(self):
        # create a standard tool
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        search_tool = Tool(
            name="search",
            func=wikipedia.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        )

        # create a custom tool
        life_tool = Tool(
            name="Meaning of Life",
            func=self.meaning_of_life,
            description="Useful for when you need to answer questions about the meaning of life. input should be MOL ",
        )

        # create a new tool
        random_tool = Tool(
            name="Random number",
            func=self.random_num,
            description="Useful for when you need to get a random number. input should be 'random'",
        )

        # create an agent
        tools = [search_tool, life_tool, random_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        # conversational agent memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=3, return_messages=True
        )

        # create our agent
        template = """
            You run in a process of Question, Thought, Action, Observation. 
            Question: The question
            Thought: Describe your thoughts about the question you have been asked.
            Observation: The result from performing the Thought
            Answer: The final answer
            
            These are all the tools available for you as an AI Assistant:

            search_tool - search_tool[input]
            life_tool - life_tool[input]
            random_tool - random_tool[input]

            However these tools should only be used if the question exactly matches the question.

            Question: {input}
            {agent_scratchpad}
        """

        prompt = ChatPromptTemplate.from_template(template)

        agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
            )
        
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def meaning_of_life(self, input=""):
        return "The meaning of life is 42 if rounded but is actually 42.17658"

    def random_num(self, input=""):
        return random.randint(0, 5)

    def run(self):
        while True:
            user_input = input(">> ")
            if user_input.lower() == "bye":
                print("LLM: Goodbye")
                break

            if user_input is not None:
                response = self.agent_executor.invoke({"input": user_input})
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
