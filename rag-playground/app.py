import os
from pprint import pprint
from colorama import Fore, Back, Style
from classes.rag_memory import RagWithMemory
from classes.qa_memory import QAMemory
from classes.CoT import COT
from classes.custom_agent import CustomAgent
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from gpt4all import GPT4All

from dotenv import load_dotenv
load_dotenv()

apiKey = os.environ["OPENAI_KEY"]
llm = ChatOpenAI(openai_api_key = apiKey, model_name = 'gpt-3.5-turbo', temperature=0)
# llm = GPT4All(model_path='./models/mistral-7b-openorca.Q4_0.gguf', n_threads=8)

def doRaqWithMemory():
    rag = RagWithMemory(llm)
    rag.loadPdf()

    print('context is loaded and ready...')

    while True:
        user_input = input('>> ')
        if user_input.lower() == 'bye':
            print('LLM: Goodbye')
            break

        if user_input is not None:
            [history, response] = rag.doLLM(user_input)
            # response = rag.chat(llm, user_input)
            # response = rag.test()
            print(Fore.RED + history)
            print(Style.RESET_ALL)

            pprint(response)

def doCOT():
    # llm = GPT4All(model='./models/mistral-7b-openorca.Q4_0.gguf', n_threads=8)

    apiKey = os.environ["OPENAI_KEY"]
    llm = ChatOpenAI(openai_api_key = apiKey, model_name = 'gpt-3.5-turbo', temperature=0.1)

    cotLLM = COT(llm)

    while True:
        user_input = input('>> ')
        if user_input.lower() == 'bye':
            print('LLM: Goodbye')
            break

        if user_input is not None:
            response = cotLLM.doLLM(user_input)
            pprint(response)

if __name__ == '__main__':
#    doRaqWithMemory()

    agent = CustomAgent(llm)
    while True:
        user_input = input('>> ')
        if user_input.lower() == 'bye':
            print('LLM: Goodbye')
            break

        if user_input is not None:
            response = agent.doLLM(user_input)
            pprint(response)
