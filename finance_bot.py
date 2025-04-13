import requests
import os

url = "https://api-inference.huggingface.co/models/gpt2"
token = os.getenv("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN not set in environment")
    exit(1)
headers = { "Authorization": f"Bearer {token}" }
question = "List 3 practical ways to save money on a tight budget"
payload = { "inputs": question, "parameters": { "max_length": 100, "num_return_sequences": 1 } }

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    data = response.json()
    print(data[0].get("generated_text", "No answer found")[:200])
else:
    print(f"Error: {response.status_code} - {response.text}")
