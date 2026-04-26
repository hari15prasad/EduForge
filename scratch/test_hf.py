import requests
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
api_url = f"https://api-inference.huggingface.co/models/{model_id}"

headers = {"Authorization": f"Bearer {token}"}

def query(payload):
    response = requests.post(api_url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    try:
        return response.json()
    except:
        print("Raw Response:", response.text)
        return {"error": "Invalid JSON"}

try:
    output = query({
        "inputs": "Can you please say 'Connectivity Test Successful'?",
    })
    print("Response:", output)
except Exception as e:
    print("Error:", e)
