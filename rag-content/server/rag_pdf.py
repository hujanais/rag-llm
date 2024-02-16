from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class RAG_PDF:
    def __init__(self, llm):
        self.vectorstore = None
        self.llm = llm
        self.isPDFLoaded = False
        self.db = None
        pass

    # Perform the LLM
    # query is the question. "How are you?"
    def doLLM(self, query):

        if self.isPDFLoaded == False:
            self.loadPDF()
            self.isPDFLoaded = True

        # Prompt Template
        template = """You are a useful assistant that can help me analyze and summarize documents. Answer questions succinctly based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        ## This is for documentation only. Example of instantiating ChatGPT or Ollama
        ## llm = ChatOpenAI(openai_api_key = self.apiKey, model_name = 'gpt-3.5-turbo', temperature=0.1)
        ## llm = Ollama(base_url='http://localhost:11434', model='llama2')
        # the llm model is passed into this class

        # load index from disk
        # https://github.com/langchain-ai/langchain/issues/7175
        # we somehow still need to set pass the embedding_function even if we are just loading
        # db = FAISS.load_local('faiss_index_pdf', GPT4AllEmbeddings())
        retriever = self.db.as_retriever()

        # Build the langchain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}   # context and question tags must match that in the PromptTemplate
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query)

        return result   

    # Retrieve Data from PDF
    # Split Data into small chunks because the LLM has a window size limitation
    # Encode the data into vectors using Embeddings
    # Save the vectorized data into a vector database like Chroma or FAISS
    def loadPDF(self):

        # load document.
        loader = PyPDFLoader("./samples/llm-proposal.pdf")
        pages = loader.load_and_split()

        # Extract text content from each page
        page_texts = [page.page_content for page in pages]

        # split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
        )
        documents = text_splitter.create_documents(page_texts)

        # reset database
        # delete all the ids one by one.
        # try:
        #     db = FAISS.load_local('faiss_index_pdf', GPT4AllEmbeddings())
        #     count = len(db.index_to_docstore_id)
        #     for _ in range(count):
        #         print(f'deleting {db.index_to_docstore_id[0]}')
        #         db.delete([db.index_to_docstore_id[0]])
        # except:
        #     print('faiss_index_pdf does not exist')

        # embed documents into vector database.
        self.db = FAISS.from_documents(documents, GPT4AllEmbeddings())
        # db.save_local('faiss_index_pdf')