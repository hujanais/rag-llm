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

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationSummaryBufferMemory)

class RagWithMemory:
    def __init__(self, llm):
        self.db = None
        self.llm = llm
        self.memory = QAMemory(6)

    def loadPdf(self):
        # load document.
        loader = WebBaseLoader("https://theleesfranceadventure.wordpress.com")
        # loader = WebBaseLoader('https://theleesnorwayadventure.wordpress.com')
        pages = loader.load_and_split()

        print(len(pages))

        # Extract text content from each page
        page_texts = [page.page_content for page in pages]

        # split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )
        documents = text_splitter.create_documents(page_texts)

        # embed documents into vector database.
        self.db = FAISS.from_documents(documents, GPT4AllEmbeddings())

    def doLLM(self, query):

        # Prompt Template
        template = """You are a useful and jovial assistant that can help me analyze and summarize documents. Answer questions based only on the following context:
        {context}

        Current conversation:
        {history}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        retriever = self.db.as_retriever()

        # Build the langchain
        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke({"question": query, "history": self.memory.getHistory()})

        # update the conversation history
        self.memory.add(query, result)

        return [self.memory.getHistory(), result]

    def chat(self, llm, query):
        resp = self.conversation.invoke(query)

        bufw_history = self.conversation.memory.load_memory_variables(
            inputs=[]
        )['history']

        print('')
        print(Fore.RED + bufw_history)
        print(Style.RESET_ALL)
        print('')

        return resp
    
    def test(self):
        vectorstore = FAISS.from_texts( ["harrison worked at kensho"], embedding=GPT4AllEmbeddings())
        retriever = vectorstore.as_retriever()

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}

        Answer in the following language: {language}
        """
        prompt = ChatPromptTemplate.from_template(template)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({"question": "where did harrison work", "language": "italian"})