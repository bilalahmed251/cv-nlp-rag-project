# 👷‍♂️ AI Visual Compliance System (CV-NLP RAG)

![AI Visual Compliance System Banner](github_banner.png)

A multi-modal Artificial Intelligence pipeline that combines **Computer Vision** and **Retrieval-Augmented Generation (RAG)** to automate safety compliance monitoring on construction sites.

## 🌟 Overview

Standard object detection models can identify objects (e.g., "person", "no_helmet"), but they lack the domain-specific context to enforce company policies. This project bridges that gap by decoupling the architecture into three core engines:

1. **The "Eyes" (CV Engine):** A custom-trained **YOLOv8** model that detects safety equipment (helmets, vests) in real-time.
2. **The "Memory" (RAG Engine):** A Retrieval-Augmented Generation system using **LangChain, HuggingFace Embeddings, and FAISS** to instantly search and retrieve the exact company safety policy from internal PDFs/Text files.
3. **The "Brain" (NLP Engine):** The **Google Gemini LLM** acts as an AI Safety Officer, taking the structured visual detections and the retrieved written policy to generate a logical, evidence-based compliance report.

Everything is wrapped in an interactive, easy-to-use **Streamlit** web application.

---

## 🛠️ Technology Stack
*   **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
*   **NLP & Generative AI:** Google Gemini (1.5 Flash)
*   **RAG Pipeline:** LangChain, HuggingFace (`all-MiniLM-L6-v2`), FAISS Vector Database
*   **User Interface:** Streamlit
*   **Language:** Python

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/bilalahmed251/cv-nlp-rag-project.git
cd cv-nlp-rag-project
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Setup API Keys
You will need a Google Gemini API Key. You can either:
- Enter the key directly into the Streamlit UI sidebar.
- Or, create a `.env` file in the root directory and add: `GEMINI_API_KEY=your_api_key_here`

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Open the local URL provided in your terminal (usually `http://localhost:8501`) to interact with the application.

---

---
*Built by Bilal Ahmed*
