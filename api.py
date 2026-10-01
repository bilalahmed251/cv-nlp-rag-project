from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import database
import shutil
from cv_engine import CVEngine
from nlp_engine import NLPEngine
from rag_engine import RAGEngine

# Global variable to hold our AI models
cv_engine = None
nlp_engine = None
rag_engine = None

# API ko batane ke liye ke Request format kaisa hoga
class AgentRequest(BaseModel):
    user_query: str
    cv_detections: list

# Initialize the FastAPI app
app = FastAPI(title="AI Visual Compliance API", version="1.0")

# CORS Middleware (Very important for Full-Stack)
# Yeh React/Next.js frontend ko allow karega ke wo is API se data le sakay
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jab server start ho, tab database initialize kar lo
@app.on_event("startup")
def startup_event():
    database.init_db()
    
    # AI Models ko RAM mein load kar lo taake API fast chale
    global cv_engine, nlp_engine, rag_engine
    print("Loading AI Engines...")
    cv_engine = CVEngine(model_name="best.pt")
    nlp_engine = NLPEngine() # Yeh khud .env se key utha lega
    rag_engine = RAGEngine()

# --- STEP 1.2: Basic Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Visual Compliance Backend API! 🚀"}

@app.get("/api/logs")
def get_incident_logs():
    """Returns the latest violation logs in JSON format for the Frontend."""
    df = database.get_recent_logs(limit=10)
    
    # Convert pandas dataframe to JSON dictionary format
    logs_list = df.to_dict(orient="records")
    return {"status": "success", "total": len(logs_list), "data": logs_list}

# Note: Hum agle steps mein yahan /analyze-video aur /ask-agent ke endpoints banayenge

# --- STEP 1.3: POST Endpoints (Connecting Engines) ---

@app.post("/api/analyze-image")
def analyze_image(file: UploadFile = File(...)):
    """Frontend se image receive karta hai aur YOLO detections return karta hai."""
    
    # 1. Image ko temporary save karna
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. YOLO (CV Engine) ko pass karna
    detections, _ = cv_engine.analyze_image(temp_path)
    
    # 3. Frontend ko JSON response bhejna
    return {
        "status": "success", 
        "filename": file.filename, 
        "detections": detections
    }

@app.post("/api/ask-agent")
def ask_agent(request: AgentRequest):
    """Frontend se query/detections lega aur RAG + Gemini se report banwayega."""
    
    # 1. RAG Engine se Policy Context nikalna
    policy_context = rag_engine.query_policies(request.user_query, top_k=3)
    
    # 2. Gemini (Brain) ko query, YOLO detections, aur policy bhej kar report lena
    report = nlp_engine.analyze_compliance(
        request.user_query, 
        request.cv_detections, 
        policy_context
    )
    
    # 3. Final Jawab Frontend ko JSON mein bhejna
    return {
        "status": "success",
        "report": report,
        "retrieved_context": policy_context
    }
