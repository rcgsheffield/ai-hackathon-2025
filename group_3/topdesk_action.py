import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Config
TOPDESK_URL = os.getenv("TOPDESK_URL")
TOKEN = os.getenv("TOPDESK_TOKEN")
INCIDENT = "SHEF 2509 7820"
FILE_PATH = "./resp.txt"

# Read the text file content
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    action_text = f.read()

    formatted_text = action_text.replace('\n', '<br>')

# Add action/memo to incident
response = requests.put(
    f"{TOPDESK_URL}/tas/api/incidents/number/{INCIDENT}",
    json={
        'action': formatted_text,
        'actionInvisibleForCaller': True
    },
    headers={
        'Authorization': f'Basic {TOKEN}',
        'Content-Type': 'application/json'
    }
)

print("✅ Action added!" if response.ok else f"❌ Failed: {response.status_code} - {response.text}")