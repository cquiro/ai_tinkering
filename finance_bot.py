import requests
import os
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

url = "https://api-inference.huggingface.co/models/gpt2"
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: HF_TOKEN not set in environment")
    exit(1)
headers = { "Authorization": f"Bearer {hf_token}" }
# question = "List 3 practical ways to save money on a tight budget for a single person earning $30,000 a year, focusing on daily expenses."
# question = "List 3 practical ways to save money on a tight budget."
# question = "Name 3 useful tips to reduce expenses with limited funds."
question = "List 3 practical ways to save money on groceries for a single person."
payload = { "inputs": question, "parameters": { "max_length": 500, "num_return_sequences": 1 } }

tokens = tokenizer.encode(question)
print(f"Prompt: {question}")
print(f"Number of tokens: {len(tokens)}")

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Bot response:")
    print(data[0].get("generated_text", "No answer found"))
else:
    print(f"Error: {response.status_code} - {response.text}")
