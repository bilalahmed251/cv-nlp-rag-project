import os
import streamlit as st
from PIL import Image
from cv_engine import CVEngine
from rag_engine import RAGEngine
from nlp_engine import NLPEngine
from dotenv import load_dotenv
import database

database.init_db()

load_dotenv()

st.set_page_config(
    page_title="AI Visual Compliance System",
    page_icon="👷‍♂️",
    layout="wide"
)

@st.cache_resource
def get_cv_engine():
    return CVEngine(model_name="yolov8n.pt")

@st.cache_resource
def get_rag_engine():
    return RAGEngine()

st.sidebar.title("⚙️ System Configuration")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key:",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Enter your Gemini API key"
)

api_key = api_key_input or os.getenv("GEMINI_API_KEY")

if api_key:
    st.sidebar.success("✅ Gemini API Key Connected!")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Vision Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.15, 
    step=0.05,
    help="Increase this if 'no helmet' is wrongly detected as 'helmet'."
)

st.sidebar.divider()
st.sidebar.subheader("📄 Dynamic Policy (RAG)")
uploaded_policy = st.sidebar.file_uploader("Upload custom safety policy", type=["pdf", "txt"])

if uploaded_policy is not None:
    if st.session_state.get('last_uploaded_policy') != uploaded_policy.name:
        with st.sidebar.status("Adding policy to Knowledge Base..."):
            temp_policy_path = os.path.join("data", uploaded_policy.name)
            os.makedirs("data", exist_ok=True)
            with open(temp_policy_path, "wb") as f:
                f.write(uploaded_policy.getbuffer())
            
            rag_engine = get_rag_engine()
            success = rag_engine.add_document(temp_policy_path)
            
            if success:
                st.session_state['last_uploaded_policy'] = uploaded_policy.name
                st.sidebar.success(f"Added {uploaded_policy.name} to RAG!")
            else:
                st.sidebar.error("Failed to process document.")
    else:
        st.sidebar.success(f"Policy '{uploaded_policy.name}' is active.")

st.title("👷‍♂️ AI Visual Compliance System")
st.markdown("""
This end-to-end system combines **Computer Vision (YOLOv8)** to detect safety gear with a 
**RAG (Retrieval-Augmented Generation)** pipeline and **Gemini LLM** to verify policy compliance.
""")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Upload Vision Data")
    uploaded_file = st.file_uploader("Upload an image or video of the worker/site", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        is_video = file_ext in ["mp4", "mov", "avi"]
        
        if not is_video:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.video(uploaded_file)
            
        st.success("File successfully loaded.")
        
        if st.button("Run CV Analysis 🔍"):
            with st.spinner("Running YOLOv8 Object Detection..."):
                cv_engine = get_cv_engine()
                
                if not is_video:
                    temp_path = "temp_uploaded.jpg"
                    image.save(temp_path)
                    
                    detections, annotated_img = cv_engine.analyze_image(temp_path, conf_threshold=conf_threshold)
                    
                    if annotated_img is not None:
                        st.image(annotated_img, caption="AI Detected Vision", use_container_width=True)
                        
                    if len(detections) == 0:
                        st.warning("No objects detected by YOLO model.")
                    else:
                        st.success("YOLO Detections Found:")
                        for det in detections:
                            st.write(f"- **{det['object']}** ({det['confidence']}% confidence)")
                        
                        st.session_state['cv_detections'] = detections
                else:
                    temp_path = "temp_uploaded." + file_ext
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    st.info("Processing video frames live...")
                    progress_bar = st.progress(0)
                    frame_placeholder = st.empty()
                    table_placeholder = st.empty()
                    final_detections = []
                    
                    for annotated_frame, current_detections, workers, progress in cv_engine.analyze_video(temp_path, conf_threshold=conf_threshold):
                        progress_bar.progress(progress)
                        if annotated_frame is not None:
                            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                            
                        if workers:
                            table_placeholder.dataframe(
                                [
                                    {
                                        "Worker ID": w["worker_id"],
                                        "Status": w["status"],
                                        "PPE Confidence": w["equipment_confidence"],
                                    }
                                    for w in workers
                                ],
                                use_container_width=True,
                                hide_index=True,
                            )
                            for w in workers:
                                database.process_worker_status(w["worker_id"], w["status"], annotated_frame)
                            
                            database.cleanup_ghosts()
                                
                        final_detections = current_detections
                        
                    progress_bar.empty()
                        
                    if len(final_detections) == 0:
                        st.warning("No objects detected in the video.")
                    else:
                        st.success("Final Aggregated Video Detections:")
                        for det in final_detections:
                            st.write(f"- **{det['object']}**")
                            
                        st.session_state['cv_detections'] = final_detections

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
                detections = st.session_state.get('cv_detections', None)
                if detections is None:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    is_video = file_ext in ["mp4", "mov", "avi"]
                    cv_engine = get_cv_engine()
                    
                    if not is_video:
                        temp_path = "temp_uploaded.jpg"
                        Image.open(uploaded_file).save(temp_path)
                        detections, _ = cv_engine.analyze_image(temp_path, conf_threshold=conf_threshold)
                    else:
                        temp_path = "temp_uploaded." + file_ext
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        for _, current_detections, _, _ in cv_engine.analyze_video(temp_path, conf_threshold=conf_threshold):
                            detections = current_detections
                            
                    st.session_state['cv_detections'] = detections
                
                rag_engine = get_rag_engine()
                policy_context = rag_engine.query_policies(user_query, top_k=3)
                
                with st.expander("📖 View Retrieved Policy Rules (RAG Context)"):
                    st.markdown(policy_context)
                
                nlp_engine = NLPEngine(api_key=api_key)
                report = nlp_engine.analyze_compliance(user_query, detections, policy_context)
                
                st.subheader("📋 AI Compliance Report")
                st.markdown(report)

st.divider()
st.header("🚨 Incident Logs & Snapshots")
st.caption("Auto-generated logs of safety violations detected during video/webcam monitoring.")

if st.button("🔄 Refresh Logs"):
    pass

logs_df = database.get_recent_logs()
if not logs_df.empty:
    st.dataframe(
        logs_df[['id', 'worker_id', 'violation_type', 'started_at', 'ended_at', 'duration_seconds', 'incident_state']], 
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No violations logged yet.")
