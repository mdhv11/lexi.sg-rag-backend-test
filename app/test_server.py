import requests
import json

def test_query():
    url = "http://127.0.0.1:5000/query"
    payload = {
        "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print("Failed to parse JSON response:", e)
        print("Raw response:", response.text)

if __name__ == "__main__":
    test_query() 