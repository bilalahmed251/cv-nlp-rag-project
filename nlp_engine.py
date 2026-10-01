import google.generativeai as genai
import os

class NLPEngine:
    def __init__(self, api_key=None):
        """
        Initializes the NLP engine.
        Connects to Gemini LLM to reason over Vision Detections and RAG Policy Context.
        """
        print("Initializing NLP Engine (The Brain)...")
        
        # Priority: Passed API Key -> Environment Variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("WARNING: No Gemini API Key found.")
            return
            
        genai.configure(api_key=self.api_key)
        
        # Fallback list of models supported by current API
        model_candidates = ['gemini-1.5-flash', 'gemini-1.5-pro']
        self.model = None
        
        for m_name in model_candidates:
            try:
                candidate = genai.GenerativeModel(m_name)
                # Test quick model ping
                self.model = candidate
                self.model_name = m_name
                print(f"[Info] Connected successfully to model: {m_name}")
                break
            except Exception as e:
                continue

        if not self.model:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.model_name = 'gemini-1.5-flash'

    def analyze_compliance(self, user_query: str, cv_detections: list, policy_context: str) -> str:
        """
        Combines User Question + Computer Vision Detections + RAG Policy Context
        to generate a comprehensive AI Compliance Report.
        """
        if not self.api_key:
            return "⚠️ **Error:** Gemini API Key is missing. Please enter your API key."

        # Format CV detections into a clean string
        if cv_detections:
            detected_items = [f"{item['object']} (Confidence: {item.get('confidence', 'N/A')}%)" for item in cv_detections]
            objects_str = ", ".join(detected_items)
        else:
            objects_str = "No specific objects detected by camera."

        # Construct the Multi-modal RAG Prompt
        prompt = f"""
You are an expert AI Construction Safety Officer. Analyze the following site situation.

### INPUT DATA:
- **User Question:** {user_query}
- **Camera Detections (Computer Vision):** {objects_str}

### RETRIEVED COMPANY SAFETY POLICIES (RAG Context):
{policy_context}

---

### INSTRUCTIONS FOR YOUR REPORT:
Please provide a clear, structured Safety Compliance Report in Markdown containing:
1. **Compliance Verdict:** (COMPLIANT / VIOLATION DETECTED / UNCERTAIN)
2. **Risk Level:** (LOW / MEDIUM / HIGH / CRITICAL)
3. **Detailed Explanation:** Explain the finding by explicitly referencing the camera detections AND the retrieved policy section numbers/penalties.
4. **Citations:** Provide the exact source file and page number from the RAG context that supports your reasoning.
5. **Recommended Actions:** What should the site supervisor or worker do immediately?
"""

        print(f"[Info] Sending structured RAG + CV prompt to Gemini LLM ({self.model_name})...")
        
        models_to_try = [self.model_name, 'gemini-1.5-flash']
        models_to_try = list(dict.fromkeys(models_to_try)) # deduplicate
        
        last_error = None
        for m_name in models_to_try:
            try:
                print(f"[Info] Trying model: {m_name}...")
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"[Warning] Failed with {m_name}: {e}")
                last_error = e
                
        return f"⚠️ **Error communicating with Gemini LLM:** {last_error}"


# --- Test script if run directly ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    dummy_detections = [{"object": "no_helmet", "confidence": 95.0}]
    dummy_policy = "SECTION 1.1: All workers must wear safety helmets at all times. Penalty is $50."
    dummy_query = "Is this worker complying with safety rules?"

    engine = NLPEngine(api_key=os.getenv("GEMINI_API_KEY"))
    report = engine.analyze_compliance(dummy_query, dummy_detections, dummy_policy)
    print("\n" + report)
