from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import tempfile
from pathlib import Path


import json
from pydantic import BaseModel
import database
from cv_engine import CVEngine
from nlp_engine import NLPEngine
from rag_engine import RAGEngine

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

cv_engine = None
nlp_engine = None
rag_engine = None

class AgentRequest(BaseModel):
    user_query: str
    cv_detections: list

class ReviewRequest(BaseModel):
    status: str # "CONFIRMED_VIOLATION" or "DISMISSED_FALSE_ALARM"
    reviewer_notes: str = ""

app = FastAPI(title="AI Visual Compliance API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    database.init_db()
    global cv_engine, nlp_engine, rag_engine
    print("Loading AI Engines...")
    try:
        cv_engine = CVEngine(model_name="best.pt")
    except Exception as e:
        print(f"[Warning] Failed to load custom model 'best.pt': {e}. Falling back to default model.")
        cv_engine = CVEngine()
    nlp_engine = NLPEngine()
    rag_engine = RAGEngine()


@app.get("/")
def read_root():
    return {"message": "Welcome to AI Visual Compliance Backend API! 🚀"}

@app.get("/api/logs")
def get_incident_logs():
    """Returns the latest violation logs in JSON format."""
    df = database.get_recent_logs(limit=10)
    logs_list = df.to_dict(orient="records")
    return {"status": "success", "total": len(logs_list), "data": logs_list}

@app.post("/api/analyze")
def analyze(file: UploadFile = File(...),
            user_query: str = Form("Is this worker compliant with the safety policy?"),
            safe_zone_json: str = Form(None),
            blur_faces: bool = Form(False)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Upload a JPG, PNG, BMP or WEBP image.")
    
    # Parse safe zones if provided (e.g. [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    safe_zones = None
    if safe_zone_json:
        try:
            parsed = json.loads(safe_zone_json)
            # Support single polygon or list of polygons
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0][0], (int, float)):
                    safe_zones = [parsed]
                else:
                    safe_zones = parsed
        except Exception as e:
            raise HTTPException(400, f"Invalid safe_zone_json format: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        detections, workers, frame = cv_engine.analyze_image_local(
            tmp.name, safe_zones=safe_zones, blur_faces=blur_faces
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp.name)

    policy_context = rag_engine.query_policies(user_query, top_k=4)
    report = nlp_engine.analyze_compliance(user_query, detections, policy_context)
    incident_ids = [
        database.log_static_incident(w["worker_id"], w["status"], frame)
        for w in workers if w["status"].startswith("VIOLATION")
    ]
    return {"status": "success", "filename": file.filename, "detections": detections,
            "workers": workers, "report": report, "incident_ids": incident_ids}

@app.get("/api/reviews/pending")
def get_pending_reviews():
    """Returns incidents currently awaiting supervisor review (Human-in-the-Loop queue)."""
    pending = database.get_pending_reviews(limit=50)
    return {"status": "success", "total_pending": len(pending), "data": pending}

@app.post("/api/reviews/{incident_id}")
def review_incident(incident_id: int, req: ReviewRequest):
    """Allows a safety officer to approve or dismiss a flagged incident."""
    valid_statuses = {"CONFIRMED_VIOLATION", "DISMISSED_FALSE_ALARM"}
    if req.status not in valid_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {valid_statuses}")
    
    updated = database.update_incident_review(incident_id, req.status, req.reviewer_notes)
    if not updated:
        raise HTTPException(404, f"Incident {incident_id} not found.")
    return {"status": "success", "incident_id": incident_id, "review_status": req.status}

@app.post("/api/ask-agent")
def ask_agent(request: AgentRequest):
    """Generates a compliance report based on query, detections, and RAG context."""
    policy_context = rag_engine.query_policies(request.user_query, top_k=4)
    
    report = nlp_engine.analyze_compliance(
        request.user_query, 
        request.cv_detections, 
        policy_context
    )
    
    return {
        "status": "success",
        "report": report,
        "retrieved_context": policy_context
    }
