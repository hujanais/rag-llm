from operator import itemgetter
from colorama import Fore, Back, Style
from classes.qa_memory import QAMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnablePassthrough, RunnableParallel)

from tavily import TavilyClient

class COT:
    def __init__(self, llm):
        self.llm = llm

    def doAgent(self):
        tavily = TavilyClient(api_key="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")
        # For basic search:
        response = tavily.search(query="Is it a holiday in the US today?")
        print(response)
        # # For advanced search:
        # response = tavily.search(query="Should I invest in Apple in 2024?", search_depth="advanced")
        # # Get the search results as context to pass an LLM:
        # context = [{"url": obj["url"], "content": obj["content"]} for obj in response.results]


        # tavily = TavilyClient(api_key="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")
        # response = tavily.search({"query": "What happened in the latest burning man floods"})
        # print(response)

    def doLLM(self, query):
        # Prompt Template
        template = """You run in a process of Question, Thought, Action, Observation. 
        Use Thought to describe your thoughts about the question you have been asked.  
        Observation will be the result of running those actions.

        If you can not answer the question from your memory, just say XXXX

        Finally at the end, state the Answer.

        Here are some sample sessions.
        Question: What is capital of france?
        Thought: This is about geography, I can recall the answer from my memory.
        Action: lookup: capital of France.
        Observation: Paris is the capital of France.
        Answer: The capital of France is Paris.

        Question: Who painted Mona Lisa?
        Thought: This is about general knowledge, I can recall the answer from my memory.
        Action: lookup: painter of Mona Lisa.
        Observation: Mona Lisa was painted by Leonardo da Vinci .
        Answer: Leonardo da Vinci painted Mona Lisa.

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Build the langchain
        chain = (
            {"question": RunnablePassthrough()} 
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke({"question": query})

        return result