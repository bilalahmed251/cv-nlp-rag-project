# 🧠 Programming Fundamentals Used in CV-NLP RAG Project

Is sheet ka maqsad aapko yeh batana hai ke aapke project mein programming ke **core (fundamental) concepts** kahan aur kaise istemal hue hain. Interviewer aksar check karta hai ke aapke basic concepts clear hain ya nahi. Is project mein aapne almost har bada programming concept touch kiya hai.

---

## 1. Object-Oriented Programming (OOP)
Aapne code ko bikhre hue functions (spaghetti code) ki tarah nahi likha, balke **Classes aur Objects** use kiye hain (Encapsulation).
*   **Kahan Use Hua?** `rag_engine.py` mein `class RAGEngine:` aur `cv_engine.py` mein `class CVEngine:`.
*   **Logic:** Har engine ki apni ek state (variables) aur behaviours (functions) hain. Jab app shuru hoti hai to `__init__` method ke zariye inka "Object" create hota hai, aur models load ho jatay hain.

## 2. Modularity (Decoupling Code)
*   **Kahan Use Hua?** Aapne sara code `app.py` mein likhne ke bajaye usko 3 alag files (`cv_engine.py`, `nlp_engine.py`, `rag_engine.py`) mein divide kiya.
*   **Logic:** Is fundamental principle ko "Separation of Concerns" (SoC) kehte hain. Agar kal ko CV model (YOLO) change karke koi aur lagana ho, to `nlp_engine.py` ya `app.py` ko change karne ki zaroorat nahi paray gi.

## 3. Data Structures (Lists & Dictionaries)
*   **Kahan Use Hua?** YOLOv8 se aane walay raw output ko parse karke List of Dictionaries mein save karna: `[{'object': 'no_helmet', 'confidence': 88}, {'object': 'person', 'confidence': 92}]`.
*   **Logic:** JSON / Dictionaries best tarika hota hai kisi bhi structured data ko ek module se dusre module (CV se NLP) bhejne ka. LLM dictionaries ko easily samajh jata hai as compared to raw text.

## 4. Control Flow (Loops & Conditionals)
*   **Kahan Use Hua?** 
    *   **Loops:** `for box in results:` (YOLO ke output array mein se har ek box ko extract karne ke liye).
    *   **Conditionals (If-Else):** `if len(detections) == 0:` (Agar CV ne kuch nahi dekha to bura error aane ke bajaye user ko pyara sa warning message dikhana).

## 5. Error Handling (Try-Except)
*   **Kahan Use Hua?** `rag_engine.py` mein jab PDF ya Text files load hoti hain:
    ```python
    try:
        pdf_loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
    ```
*   **Logic:** Agar folder mein koi file corrupt hai ya gayab hai, to poori app crash hone ke bajaye sirf error throw karegi aur baqi app chalti rahegi.

## 6. State Management
*   **Kahan Use Hua?** `app.py` mein Streamlit ke andar `st.session_state`.
*   **Logic:** Web browsers bhool jatay hain ke pichli screen par kya tha. Jab aapne "Analyze Compliance" click kiya to page refresh hota hai. Humne session state use kiya taa ke pichli step ki tasweer (CV detections) memory se delete na ho jaye.

## 7. Caching & Performance Optimization (Decorators)
*   **Kahan Use Hua?** `app.py` mein humne `@st.cache_resource` use kiya.
*   **Logic:** AI models (YOLO, HuggingFace Embeddings) GBs mein RAM letay hain aur load hone mein time lagatay hain. Caching ka concept yeh hai ke inko sirf zindgi mein ek dafa (first run par) memory mein load karo, aur baar baar har click par dubara load hone se bacha lo (taake app fast ho jaye).

---
*Yeh sheet aapki personal interview preparation ke liye hai.*
