import os
import json
from mistralai.client import MistralClient
import pandas as pd
import functools
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient

from dotenv import load_dotenv
load_dotenv()

apiKey = os.environ["MISTRAL_API_KEY"]
llm = MistralClient(api_key=apiKey)

# Assuming we have the following data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

# Create DataFrame
df = pd.DataFrame(data)

def retrieve_payment_status(df: pd.DataFrame, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df: pd.DataFrame, transaction_id: str) -> str: # type: ignore
    if transaction_id in df.transaction_id.values: 
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]

names_to_functions = {
  'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
  'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}

func = names_to_functions['retrieve_payment_status']
print(func(transaction_id='T1001', df=df))

messages = [
    ChatMessage(role="user", content="What's the status of my transaction?")
]

model = 'mistral-large-latest'

response = llm.chat(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

messages.append(ChatMessage(role="assistant", content=response.choices[0].message.content))
messages.append(ChatMessage(role="user", content="My transaction ID is T1001."))

response = llm.chat(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

messages.append(response.choices[0].message)

for message in messages:
    print(message)

tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)
print("\nfunction_name: ", function_name, "\nfunction_params: ", function_params)

function_result = names_to_functions[function_name](**function_params) # ** to unpack from dictionary.
messages.append(ChatMessage(role="tool", name=function_name, content=function_result))

response = llm.chat(
    model=model,
    messages=messages
)

print (response.choices[0].message)