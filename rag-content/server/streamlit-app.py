from collections import deque
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import streamlit as st
from rag_pdf import RAG_PDF

class StreamLitRag():
    def __init__(self):
        print('init')
        apiKey = os.environ["OPENAI_KEY"]
        PORT = int(os.environ.get('PORT', 5000))
        llm = ChatOpenAI(openai_api_key = apiKey, model_name = 'gpt-3.5-turbo', temperature=0.1)
        self.ragPdf = RAG_PDF(llm)
        self.messages = deque()

    def build_streamLit(self):
        # Set page title and description
        st.title("Interactive Q&A App")
        st.write("Type your question below and get the answer.")

        # User input field for questions
        user_question = st.text_input("Your Question:")

        # Button to trigger the response
        if st.button("Submit"):
            # Get the answer based on the user's question
            # answer = self.get_answer(user_question)
            answer = 'fubar'
            self.messages.append({"Q": user_question, "A": answer})
            print(len(self.messages))
            

            # Display the answer in a scrollable area
            # answer = '/n'.join(self.messages)
            st.text_area("Answer:", answer, height=400)

    def get_answer(self, query: str):
        response = self.ragPdf.doLLM(query)
        return f"{response}"

if __name__ == "__main__":
    streamLitRag = StreamLitRag()
    streamLitRag.build_streamLit()
