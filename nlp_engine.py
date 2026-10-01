import os
from google import genai

import time
import random
from google.genai import errors

class NLPEngine:
    def __init__(self, api_key=None):
        print("Initializing NLP Engine (The Brain)...")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("WARNING: No Gemini API Key found.")
            self.client = None
            return
            
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(f"[Info] Connected successfully to Google GenAI client.")
            print("[Info] Available Models:")
            try:
                for m in self.client.models.list():
                    if 'gemini' in m.name.lower():
                        print(f"  - {m.name}")
            except Exception as e:
                print(f"[Warning] Could not list models: {e}")
        except Exception as e:
            print(f"[Error] Failed to initialize GenAI client: {e}")
            self.client = None

    def _generate(self, prompt: str, retries: int = 3) -> str:
        primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
        
        models_to_try = []
        if primary_model: models_to_try.append(primary_model)
        if fallback_model and fallback_model not in models_to_try: models_to_try.append(fallback_model)
        
        last_error = None
        for model in models_to_try:
            for attempt in range(retries + 1):
                try:
                    response = self.client.models.generate_content(model=model, contents=prompt)
                    return response.text
                except errors.APIError as e:
                    last_error = e
                    if e.code == 404:
                        print(f"[Warning] Model {model} not found (404). Skipping to next model.")
                        break # Skip to next model
                    elif e.code in [429, 500, 503, 504]:
                        if attempt < retries:
                            sleep_time = (2 ** attempt) + random.random()
                            print(f"[Warning] API error {e.code} on {model}. Retrying in {sleep_time:.2f}s...")
                            time.sleep(sleep_time)
                        else:
                            print(f"[Error] Max retries reached for model {model} due to error {e.code}.")
                    else:
                        raise e # Other API errors
                        
        if last_error:
            raise last_error
        raise Exception("All models and retries failed.")

    def analyze_compliance(self, user_query: str, cv_detections: list, policy_context: str) -> str:
        if not self.client:
            return "⚠️ **Error:** Gemini API Key is missing or client failed to initialize."

        if cv_detections:
            detected_items = [f"{item['object']} (Confidence: {item.get('confidence', 'N/A')}%)" for item in cv_detections]
            objects_str = ", ".join(detected_items)
            
            prompt = f"""
You are an expert AI Construction Safety Officer. Analyze the following site situation based on the camera detections.

### INPUT DATA:
- **User Question:** {user_query}
- **Camera Detections:** {objects_str}

### RETRIEVED COMPANY SAFETY POLICIES (RAG Context):
{policy_context}

---
### INSTRUCTIONS FOR YOUR REPORT:
Please provide a clear, structured Safety Compliance Report in Markdown containing:
1. **Compliance Verdict:** (COMPLIANT / VIOLATION DETECTED / UNCERTAIN)
2. **Risk Level:** (LOW / MEDIUM / HIGH / CRITICAL)
3. **Detailed Explanation:** Explain the finding by explicitly referencing the camera detections AND the retrieved policy.
4. **Citations:** Provide the exact source file from the RAG context.
5. **Recommended Actions:** What should the site supervisor or worker do immediately?
"""
        else:
            prompt = f"""
You are an expert AI Construction Safety Officer. The user is asking a general safety policy question.

### INPUT DATA:
- **User Question:** {user_query}

### RETRIEVED COMPANY SAFETY POLICIES (RAG Context):
{policy_context}

---
### INSTRUCTIONS:
Answer the user's question directly and professionally using ONLY the provided RAG Context. 
Do not ask for camera detections. Just explain the safety rule, penalty, or procedure as requested.
If the context doesn't contain the answer, politely state that it's not in the current safety policies.
"""

        print(f"[Info] Sending structured RAG prompt to Gemini LLM...")
        
        try:
            return self._generate(prompt)
        except Exception as e:
            print(f"[Error] Gemini generation failed: {e}")
            return f"The AI model is temporarily busy. Here are the relevant policy excerpts:\n\n{policy_context}"

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    dummy_detections = [{"object": "no_helmet", "confidence": 95.0}]
    dummy_policy = "SECTION 1.1: All workers must wear safety helmets at all times. Penalty is $50."
    dummy_query = "Is this worker complying with safety rules?"

    engine = NLPEngine(api_key=os.getenv("GEMINI_API_KEY"))
    report = engine.analyze_compliance(dummy_query, dummy_detections, dummy_policy)
    print("\n" + report)
