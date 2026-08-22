import os
import streamlit as st
from PIL import Image
from cv_engine import CVEngine
from rag_engine import RAGEngine
from nlp_engine import NLPEngine

# Set page configuration for a professional look
st.set_page_config(
    page_title="AI Visual Compliance System",
    page_icon="👷‍♂️",
    layout="wide"
)

# --- Caching Heavy Models ---
@st.cache_resource
def get_cv_engine():
    return CVEngine(model_name="best.pt")

@st.cache_resource
def get_rag_engine():
    return RAGEngine()

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ System Configuration")
default_key = "AIzaSyB09-fkjtibmicAeFS6vyTLAecghJjBGB0"
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key:",
    value=default_key,
    type="password",
    help="Gemini API key configured for testing"
)

# Use entered API key or fallback to default
api_key = api_key_input or os.getenv("GEMINI_API_KEY") or default_key

if api_key:
    st.sidebar.success("✅ Gemini API Key Connected!")

# --- App Title and Description ---
st.title("👷‍♂️ AI Visual Compliance System")
st.markdown("""
This end-to-end system combines **Computer Vision (YOLOv8)** to detect safety gear with a 
**RAG (Retrieval-Augmented Generation)** pipeline and **Gemini LLM** to verify policy compliance.
""")
st.divider()

# --- Main 2-Column Layout ---
col1, col2 = st.columns([1, 1])

# --- Column 1: Image Upload & CV Detection ---
with col1:
    st.header("1. Upload Vision Data")
    uploaded_file = st.file_uploader("Upload an image of the worker/site", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("Image successfully loaded.")
        
        if st.button("Run CV Analysis 🔍"):
            with st.spinner("Running YOLOv8 Object Detection..."):
                # Save temporary image for OpenCV/YOLO
                temp_path = "temp_uploaded.jpg"
                image.save(temp_path)
                
                # Run Computer Vision Engine
                cv_engine = get_cv_engine()
                detections = cv_engine.analyze_image(temp_path)
                
                # Display Results
                if len(detections) == 0:
                    st.warning("No objects detected by YOLO model.")
                else:
                    st.success("YOLO Detections Found:")
                    for det in detections:
                        st.write(f"- **{det['object']}** ({det['confidence']}% confidence)")
                    
                    # Store detections in session state
                    st.session_state['cv_detections'] = detections

# --- Column 2: RAG + LLM Compliance Query ---
with col2:
    st.header("2. AI Compliance Reasoning")
    user_query = st.text_input(
        "Ask a safety question:",
        value="Is this worker complying with safety protocols?"
    )
    
    if st.button("Analyze Compliance 🧠"):
        if uploaded_file is None:
            st.error("Please upload an image first!")
        else:
            with st.spinner("Running RAG Retrieval & Gemini LLM Reasoning..."):
                # Step 1: Ensure CV analysis has been performed
                detections = st.session_state.get('cv_detections', None)
                if detections is None:
                    # Automatically run CV Engine if user clicked Analyze directly
                    temp_path = "temp_uploaded.jpg"
                    image.save(temp_path)
                    cv_engine = get_cv_engine()
                    detections = cv_engine.analyze_image(temp_path)
                    st.session_state['cv_detections'] = detections
                
                # Step 2: RAG Engine - Retrieve Relevant Policy Chunks
                rag_engine = get_rag_engine()
                policy_context = rag_engine.query_policies(user_query, top_k=3)
                
                # Display Retrieved Policy Context in Expander
                with st.expander("📖 View Retrieved Policy Rules (RAG Context)"):
                    st.markdown(policy_context)
                
                # Step 3: LLM Reasoning - Generate Final Report
                nlp_engine = NLPEngine(api_key=api_key)
                report = nlp_engine.analyze_compliance(user_query, detections, policy_context)
                
                # Step 4: Display Final Compliance Report
                st.subheader("📋 AI Compliance Report")
                st.markdown(report)
