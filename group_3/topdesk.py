import requests
from dotenv import load_dotenv
import os
load_dotenv()

# Config
TOPDESK_URL = os.getenv("TOPDESK_URL")
TOKEN = os.getenv("TOPDESK_TOKEN")
INCIDENT = f"SHEF%202509%207820"
FILE_PATH = "./resp.json"

# Upload with invisible flag
with open(FILE_PATH, 'rb') as f:
    response = requests.post(
        f"{TOPDESK_URL}/tas/api/incidents/number/{INCIDENT}/attachments",
        files={'file': f},
        data={'invisibleForCaller': 'true'},  # Hide from caller
        headers={'Authorization': f'Basic {TOKEN}'}
    )
    
print("✅ Uploaded!" if response.ok else f"❌ Failed: {response.text}")