import os, sys, base64, requests
from dotenv import load_dotenv

load_dotenv()
path = sys.argv[1]
with open(path, "rb") as f:
    img = base64.b64encode(f.read()).decode("utf-8")

for v in [10]:
    r = requests.post(
        f"https://serverless.roboflow.com/hard-hat-detection-4iltu/{v}",
        params={"api_key": os.getenv("ROBOFLOW_API_KEY"), "confidence": 10},
        data=img,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    print(f"--- version {v}: status {r.status_code}")
    if r.status_code == 200:
        for p in r.json().get("predictions", []):
            print(p["class"], round(p["confidence"] * 100, 1))
    else:
        print(r.text[:150])
