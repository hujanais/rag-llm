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

        tool_list = (
            render_text_description_and_args(tools)
            .replace("{", "{{")
            .replace("}", "}}")
        )
        tool_names = [t.name for t in tools]

        template = f"""Answer the following questions as best you can.
            You can answer directly if the user is greeting you or similar.
            Otherise, you have access to the following tools:

            {tool_list}

            The way you use the tools is by specifying a json blob.
            Specifically, this json should have a `action` key (with the name of the tool to use)
            and a `action_input` key (with the input to the tool going here).
            The only values that should be in the "action" field are: [{tool_names}]
            The $JSON_BLOB should only contain a SINGLE action, 
            do NOT return a list of multiple actions.
            Here is an example of a valid $JSON_BLOB:
            ```
            {{{{
                "action": $TOOL_NAME,
                "action_input": $INPUT
            }}}}
            ```
            The $JSON_BLOB must always be enclosed with triple backticks!

            ALWAYS use the following format:
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action:```
            $JSON_BLOB
            ```
            Observation: the result of the action... 
            (this Thought/Action/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin! Reminder to always use the exact characters `Final Answer` when responding.'
        """
        # prompt = ChatPromptTemplate.from_template(template=template)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    template,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm_with_stop = llm.bind(stop=["\nObservation"])
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_stop
            | ReActJsonSingleInputOutputParser()
        )

        # zero_shot_agent_chain = (
        #     RunnablePassthrough.assign(
        #         agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        #     )
        #     | prompt
        #     | llm_with_stop
        #     | ReActSingleInputOutputParser()
        # )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )

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
