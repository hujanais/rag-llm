import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
load_dotenv()

from rag import RAG
from rag_pdf import RAG_PDF

from flask import Flask, request, jsonify

app = Flask(__name__)

# Create the llm of your choice.
# ChatGPT
apiKey = os.environ["OPENAI_KEY"]
PORT = int(os.environ.get('PORT', 5000))
llm = ChatOpenAI(openai_api_key = apiKey, model_name = 'gpt-3.5-turbo', temperature=0.1)

# Ollama
# llm = Ollama(base_url='http://localhost:11434', model='llama2')

raq = RAG(llm)
raqPdf = RAG_PDF(llm)

# curl --location 'http://127.0.01:5000/api/context' --data ["https://hujanais.github.io", "https://hujanais.github.io/edge-llm"]
# request-body : [ "1", "2", "3", "4"]
@app.route('/api/context', methods=['POST'])
def create_context():
    try:
        urls = request.get_json()
        raq.loadHTML(urls)

        return jsonify({'data': 'context-created', 'error': None})
    except Exception as e:
        return jsonify({'data': None, 'error': f'{e}'})

# curl --location 'http://127.0.01:5000/api/llm' --data '{ "query": "Can you summarize some of the technologies mentioned?" }'
# request-body: { query: "What is the major topic that the author is writing about?"}
@app.route('/api/llm', methods=['POST'])
def doLLM():
    try:
        # get the user query
        # { query : "Given the context, can you tell me what are some of the main topics discussed in the website."
        json = request.get_json()
        query = json['query']

        response = raq.doLLM(query)
        
        return jsonify({'data': response, 'error': None})
    except Exception as e:
        return jsonify({'data': None, 'error': f'{e}'})

# curl --location 'http://127.0.01:5000/api/llm-pdf' --data '{ "query": "Can you summarize some of the technologies mentioned?" }'
# request-body: { query: "What is the major topic that the author is writing about?"}
@app.route('/api/llm-pdf', methods=['POST'])
def doLLM_PDF():
    try:
        # get the user query
        # { query : "Given the context, can you tell me what are some of the main topics discussed in the website."
        json = request.get_json()
        query = json['query']

        response = raqPdf.doLLM(query)
        
        return jsonify({'data': response, 'error': None})
    except Exception as e:
        return jsonify({'data': None, 'error': f'{e}'})

@app.route('/api', methods=['GET'])
def doLLM_Test():
    return jsonify({'data': 'hello', 'error': None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
