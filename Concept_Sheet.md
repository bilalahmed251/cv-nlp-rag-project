# Project Architecture Breakdown: Concept → Why → How → Code

Yeh sheet aapke interview ya concept revision ke liye design ki gayi hai. Har module ko 4 hisson mein divide kiya gaya hai: **Concept (Kya hai?)**, **Why (Kyun use kiya?)**, **How (Kaise kaam karta hai?)**, aur **Code (Main snippet)**.

---

## 1. RAG Engine (The "Memory")
File: `rag_engine.py`

*   **Concept:** RAG (Retrieval-Augmented Generation) ek aesa system hai jo aapke private data (jaise company policies) ko AI ke database mein convert karta hai, taa ke AI un policies ko padh kar answer de sake.
*   **Why:** LLMs (jaise Gemini) ko general knowledge hoti hai, lekin unhein aapki company ke specific rules nahi pata hote (e.g., helmet na pehnne par kya fine hai). RAG model ko hallucinate (galat jawabaat dene) se rokti hai aur accurate reference deti hai.
*   **How:** 
    1. **LangChain** ke zariye PDFs ko load karke chotey text chunks mein toda (split) jata hai.
    2. **HuggingFace (`all-MiniLM-L6-v2`)** un text chunks ko vectors (numbers) mein convert karta hai (Embeddings).
    3. **FAISS** in vectors ko store karta hai taa ke jab koi query aaye to wo milliseconds mein sabse relevant rule dhund kar nikal le.
*   **Code (Core Logic):**
```python
# How text is converted to vectors and stored in FAISS database
def build_faiss_index(chunks, embeddings):
    # Vector store in-memory database banata hai
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Searching for the exact policy rule
results = vector_store.similarity_search("Helmet policy", k=3)
```

---

## 2. Computer Vision Engine (The "Eyes")
File: `cv_engine.py` / `train_custom_model.py`

*   **Concept:** Yeh module images ya camera feed ko analyze karke objects (insaan, helmet, vest) ko detect karta hai aur unki screen par location (bounding boxes) identify karta hai.
*   **Why:** AI pipeline ko visual data ko samajhne ke liye ek sensor chahiye. Hum classification ke bajaye Object Detection use kar rahe hain taa ke pata chal sake ke "kon kahan hai" aur "kya usne safety gear pehna hai".
*   **How:** Humne **YOLOv8** (You Only Look Once) architecture use kiya hai kyunke yeh real-time processing ke liye sabse best hai. Model ko custom dataset (Roboflow) par train kiya gaya hai jisme safety gears mark kiye gaye the.
*   **Code (Core Logic):**
```python
from ultralytics import YOLO

# Load custom trained weights
model = YOLO('best.pt')

# Run inference to detect objects
results = model(image_path)

# Extracting detected objects into structured data for NLP
detected_items = []
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    detected_items.append(class_name) 
    # Output example: ['person', 'no_helmet']
```

---

## 3. NLP Engine (The "Brain")
File: `nlp_engine.py`

*   **Concept:** CV Engine sirf objects batata hai, faisla nahi karta. NLP Engine in objects ka context samajh kar human-like safety report ya alert generate karta hai.
*   **Why:** Code ke zariye hardcoded if-else statements (e.g., `if 'person' and 'no_helmet': alert()`) banana scalable nahi hai. LLMs complex situations ko samajh sakte hain aur properly formatted report bana sakte hain.
*   **How:** Hum **Google Gemini API** ka istemal karte hain. CV engine se milne walay raw data (arrays) aur RAG engine se milne wali company policy ko mila kar ek prompt banaya jata hai aur Gemini ko bheja jata hai. Gemini as a "Safety Officer" usko analyze karta hai.
*   **Code (Core Logic):**
```python
import google.generativeai as genai

# Setup LLM Model
model = genai.GenerativeModel('gemini-1.5-flash')

# Combining CV data and RAG policy context
prompt = f"""
You are a Safety Officer.
Detected Objects: {detected_items}
Company Policy: {policy_context}
Is there a safety violation? Generate a report.
"""

# Generating intelligent response
response = model.generate_content(prompt)
print(response.text)
```
