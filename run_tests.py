import os
import sys
from dotenv import load_dotenv

# Load .env explicitly for tests
load_dotenv()

# Add Phase_3_Backend to path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Phase_3_Backend')))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def run_tests():
    print("===== Running End-to-End Tests for RAG Application =====")
    
    # 1. Check if .env is loaded and GROQ API KEY is present
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ [ENV TEST] GROQ_API_KEY successfully loaded from environment.")
        print(f"    Key Starts With: {groq_key[:6]}...")
    else:
        print("❌ [ENV TEST] GROQ_API_KEY NOT found in environment. Ensure .env is loaded correctly.")
        
    print("\n[API TEST] Hitting the FastAPI /chat endpoint to test LLM & Vector DB integration...")
    payload = {
        "query": "What is the price of UPSC Prarambh?"
    }
    
    try:
        response = client.post("/chat", json=payload)
        if response.status_code == 200:
            print("✅ [API TEST] Endpoint successfully returned 200 OK")
            res_json = response.json()
            answer = res_json.get("answer", "")
            
            print(f"\n[RESPONSE PREVIEW]:\n{answer}\n")
            
            if res_json.get("fallback", False):
                print("⚠️ [RESULT] API returned a fallback response (LLM bypassed).")
            else:
                if len(answer) > 10:
                    print("✅ [RESULT] Successful! LLM accurately generated a response via Groq.")
                else:
                    print("❌ [RESULT] Failed: Output answer was extremely short or empty.")
        else:
            print(f"❌ [API TEST] Failed: API returned status {response.status_code}")
            print(f"Details: {response.text}")
    except Exception as e:
        print(f"❌ [API TEST] Failed: Exception occurred during request.")
        print(e)
        
    print("\n===== End of Tests =====")

if __name__ == "__main__":
    run_tests()
