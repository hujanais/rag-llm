from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

class RagWithMemory:
    def __init__(self, llm):
        self.db = None
        self.llm = llm

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
        template = """You are a useful assistant that can help me analyze and summarize documents. Answer questions succinctly based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        retriever = self.db.as_retriever()

        # Build the langchain
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }  # context and question tags must match that in the PromptTemplate
            | prompt
            | self.llm
            | StrOutputParser()
        )

        from langchain.chains.conversation.memory import ConversationSummaryMemory

        conversation = ConversationChain(
            llm=self.llm, memory=ConversationBufferWindowMemory(k=2),
            verbose=True
        )

        chain_with_memory = chain | conversation

        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # chain = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=retriever, memory=memory, verbose=True)

        # question_generator_chain = LLMChain(llm=self.llm, prompt=prompt)
        # chain = ConversationalRetrievalChain.from_llm(
        #     llm=self.llm,
        #     retriever=retriever,
        #     question_generator=question_generator_chain,
        # )

        # return chain.invoke({'question': query})

        # conversation_with_summary = ConversationChain(
        #     prompt=prompt,
        #     llm=self.llm,
        #     # We set a low k=2, to only keep the last 2 interactions in memory
        #     memory=ConversationBufferWindowMemory(memory_key='chat_history' k=2),
        #     verbose=True
        # )

        # return conversation_with_summary.predict(input=query)

        result = chain_with_memory.invoke(query)

        return result
