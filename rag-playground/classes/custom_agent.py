from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents import Tool

class CustomAgent:
    def __init__(self, llm):
        self.llm = llm

    # def doLLM2(self, query: str):
    #     template = """
    #         You are a friendly and helpful assistant that run in a process of Question, Thought, Action, Observation.
    #         TOOLS:
    #         ------

    #         Assistant has access to the following tools:

    #         add_2_numbers - [a,b]
    #         get_word_length - word
    #         turn_lights_on - light_id
    #         turn_lights_off - light_id

    #         To use a tool, please use the following format:

    #         ```
    #         Thought: Do I need to use a tool? Yes
    #         Action: the action to take, should be one of ['get_word_length', 'add_2_numbers, 'turn_lights_on', 'turn_lights_off']
    #         Action Input: the input to the action
    #         Observation: the result of the action
    #         ```

    #         When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    #         ```
    #         Thought: Do I need to use a tool? No
    #         Final Answer: [your response here]
    #         ```

    #         Begin!

    #         New input: {input}
    #         {agent_scratchpad}
    #     """

    #     prompt = ChatPromptTemplate.from_template(template=template)

    #     tools = [
    #         self.get_word_length,
    #         self.add_2_numbers,
    #         self.turn_lights_off,
    #         self.turn_lights_on,
    #     ]
    #     llm_with_tools = self.llm.bind_tools(tools)

    #     agent = (
    #         {
    #             "input": lambda x: x["input"],
    #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(
    #                 x["intermediate_steps"]
    #             ),
    #         }
    #         | prompt
    #         | llm_with_tools
    #         | OpenAIToolsAgentOutputParser()
    #     )

    #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    #     resp = list(agent_executor.stream({"input": query}))
    #     return resp

    # def doLLM3(self, query: str):
    #     # Only if you explicitly asked to get the length of a word or to add 2 numbers, use Action to run one of these actions available to you:
    #     template = """
    #         You run in a process of Question, Thought, Action, Observation.

    #         Use Thought to describe your thoughts about the question you have been asked.
    #         Observation will be the result of running those actions.

    #         Before you answer, determine whether the question is best answered using the Actions that are available to you.
            
    #         length - get_word_length
    #         add - add_2_numbers

    #         Finally at the end, state the Answer.

    #         Here are some sample sessions.

    #         Question: How many characters in the word, mombasa?
    #         Thought: This is related to getting the length of a word and I always use word_count action.
    #         Action: get_word_length: mombasa
    #         Observation: 7
    #         Answer: The length is 7

    #         Question: What is the length of the word, "ABCD"?
    #         Thought: This is related to getting the length of a word and I always use word_count action.
    #         Action: get_word_length: ABCD
    #         Observation: 4
    #         Answer: The length is 4

    #         Question: Hi. My name is Rachael?
    #         Thought: This is not related to getting the length of a word so I will just answer the question in my friendly voice.
    #         Action: No Action to Invoke
    #         Observation: Hello Rachael.  Nice to meet you.  How can I help you today?
    #         Answer: Hello Rachael.  Nice to meet you.  How can I help you today?


    #         Question: {input}

    #     """
    #     prompt = ChatPromptTemplate.from_template(template=template)

    #     tools = [self.get_word_length, self.add_2_numbers]
    #     llm_with_tools = self.llm.bind_tools(tools)

    #     agent = (
    #         {
    #             "input": lambda x: x["input"],
    #             "agent_scratchpad": lambda x: format_to_openai_tool_messages(
    #                 x["intermediate_steps"]
    #             ),
    #         }
    #         | prompt
    #         | llm_with_tools
    #         | OpenAIToolsAgentOutputParser()
    #     )

    #     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    #     resp = list(agent_executor.stream({"input": query}))
    #     return resp

    def doLLM(self, query: str):
        template = """
            You are a friendly and helpful assistant that run in a process of Question, Thought, Action, Observation.

            Use Thought to describe your thoughts about the question you have been asked.
            Observation will be the result of running those actions.
            Thought will describe your thoughts about the question you have been asked.
            
            These actions are available to you but you will only use them if they are explicitly requested or if they are highly related to the actions:
            These are the Actions that are available to you:
                   
            add_2_numbers - [a,b]
            get_word_length - word
            turn_lights_on - light_id
            turn_lights_off - light_id
            
            When asked a question, determine whether the question can be best answered by a tool.
            
            To use a tool, please use the following format:

            ```
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of ['get_word_length', 'add_2_numbers, 'turn_lights_on', 'turn_lights_off', 'meaning_of_life']
            Action Input: the input to the action
            Observation: the result of the action
            ```

            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

            ```
            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
            ```

            Finally at the end, state the Answer.

            Question: {input}
        """
        prompt = ChatPromptTemplate.from_template(template=template)

        tools = [
            self.get_word_length,
            self.add_2_numbers,
            self.turn_lights_off,
            self.turn_lights_on,
        ]
        llm_with_tools = self.llm.bind_tools(tools)

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

        agent_executor = AgentExecutor(agent=agent, tools=tools)
        resp = list(agent_executor.stream({"input": query}))
        return resp

    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        # return len(word)
        return 999

    @tool
    def add_2_numbers(a: int, b: int) -> int:
        """Returns the addition of 2 numbers."""
        # return a + b
        return 999

    @tool
    def turn_lights_on(lightId: str) -> str:
        """Turns the lights on and return status"""
        resp = f"Turn {lightId} lights on"
        return f'{"lightId":"{lightId}", "action": "on"}'

    @tool
    def turn_lights_off(lightId: str) -> str:
        """Turns the lights off and return status"""
        resp = f"Turn {lightId} lights off"
        return f'{"lightId":"{lightId}", "action": "off"}'
