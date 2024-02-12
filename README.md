## Table of Contents
|project  |description  |
|--|--|
|rag-content  |Introduction to using RAQ with Langchain  |

### Guide for raq-content project
I highly recommend a Python virtual environment for your work
```
pip3 install virtualenv
cd /rag-llm
virtualenv venv
source venv/bin/activate	# activate virtual environment to start work
cd /rag-content
pip3 install -r requirements.txt

# rename .env-template file as .env
# and enter your OpenAI api key
OPENAI_KEY=sk-#####################

# start the Flask development server
python3 server/app.py

# To test, you can use Postman or just curl from terminal.
# This sample code uses html links as import.  You can update it to 
# read local documents.
Step 1.  Feed the content links
curl --location 'http://127.0.01:5000/api/context' \
--data  '[
"https://hujanais.github.io", 
"https://hujanais.github.io/edge-llm",
"https://hujanais.github.io/edge-llm/part-1",
"https://hujanais.github.io/edge-llm/part-2",
""https://hujanais.github.io/edge-llm/part-3"
]'

Step 2. Query the documents with LLM
curl --location 'http://127.0.01:5000/api/llm' \
--data  '{ "query": "Can you summarize some of the technologies mentioned?" }'

deactivate # destroy virtual environment when done
```
### Running with Docker
There server has also been setup with Docker support for easier deployment.
```
docker run -d -p 5000:5000 -e OPENAI_KEY=YOUR-OPENAI_API_KEY --name rag-llm [IMAGEID]
```

> Written with [StackEdit](https://stackedit.io/).