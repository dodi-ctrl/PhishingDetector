from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import EmailRequest, EmailResponse
from email import message_from_string
from email.policy import default as email_default_policy
from fastapi.middleware.cors import CORSMiddleware
import base64
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
sys.path.append(os.path.dirname(__file__))

from url_agent import URLAgent
from metadata_agent import MetadataAgent
from core.feature_extraction import FeatureExtractor



# -------------------------------------------------------
# Global agents store
# -------------------------------------------------------
agents = {}
CLASS_NAMES = ["Safe", "Phishing"]

# -------------------------------------------------------
# Email parser — copied from gradio_demo_cell.py
# -------------------------------------------------------
def parse_eml_text(raw_bytes: bytes) -> tuple[str, str]:
    try:
        msg = message_from_string(
            raw_bytes.decode("utf-8", errors="ignore"),
            policy=email_default_policy
        )
    except Exception:
        msg = message_from_string(
            raw_bytes.decode("latin-1", errors="ignore"),
            policy=email_default_policy
        )

    subject = str(msg.get("Subject", "") or "")
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_content()
                except Exception:
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                break
        if not body:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        body = part.get_content()
                    except Exception:
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
    else:
        try:
            body = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="ignore") if payload else str(msg.get_payload())

    return subject, body

# -------------------------------------------------------
# DistilBERT inference — mirrors gradio_demo_cell.py
# -------------------------------------------------------
def predict_proba_for_lime(texts):
    model   = agents["text_model"]
    tokenizer = agents["tokenizer"]
    device  = agents["device"]
    

    all_probs = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

# -------------------------------------------------------
# App setup — load all models once at startup
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    # Random Forest agents
    url = URLAgent()
    url.load_model("models/url_agent.pkl")
    agents["url"] = url
    print("URL agent loaded.")

    meta = MetadataAgent()
    meta.load_model("models/metadata_agent.pkl")
    agents["metadata"] = meta
    print("Metadata agent loaded.")

    # DistilBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained("models/distilbert_phishing")
    text_model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_phishing")
    text_model.to(device)
    text_model.eval()

    agents["tokenizer"]   = tokenizer
    agents["text_model"]  = text_model
    agents["device"]      = device
    agents["lime"]        = LimeTextExplainer(class_names=CLASS_NAMES)
    print(f"DistilBERT loaded on {device}.")

    print("All models ready.")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="Phishing Detector API",
    description="Multi-agent phishing detection: DistilBERT + URL RF + Metadata RF with LIME.",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Phishing Detector API is running"}


@app.post("/analyse", response_model=EmailResponse)
def analyse_email(req: EmailRequest):

    # 1. Decode base64
    try:
        raw_bytes = base64.b64decode(req.raw_email_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    # 2. Parse email text
    subject, body = parse_eml_text(raw_bytes)
    full_text = f"Subject: {subject}\n\n{body}".strip()

    if not full_text:
        raise HTTPException(status_code=422, detail="Could not extract text from email")

    # 3. DistilBERT + LIME
    # 3. DistilBERT
    try:
        probs = predict_proba_for_lime([full_text])[0]
        p_safe, p_phish = float(probs[0]), float(probs[1])
        text_verdict = CLASS_NAMES[int(np.argmax(probs))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text agent failed: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text agent failed: {str(e)}")

    # 4. URL + Metadata RF agents
    try:
        extractor = FeatureExtractor()
        features  = extractor.extract_features_from_eml(raw_bytes)
        url_result  = agents["url"].get_prediction_with_confidence(features)
        meta_result = agents["metadata"].get_prediction_with_confidence(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RF agents failed: {str(e)}")

    # 5. Weighted fusion (based on README accuracy scores)
    #    Text: 97.34% → 0.30 | URL: 99.34% → 0.35 | Metadata: 99.92% → 0.35
    fused = (
        0.30 * p_phish +
        0.35 * url_result["phishing_probability"] +
        0.35 * meta_result["phishing_probability"]
    )
    final_verdict = "phishing" if fused >= 0.5 else "legitimate"

    return {
        "verdict": final_verdict,
        "confidence": round(fused, 4),
        "agents": {
            "text": {
                "verdict": text_verdict,
                "phishing_probability": round(p_phish, 4),
                "safe_probability": round(p_safe, 4),
                "confidence": round(max(p_safe, p_phish), 4),
                
            },
            "url": {
                "verdict": url_result["verdict"],
                "phishing_probability": round(url_result["phishing_probability"], 4),
                "legitimate_probability": round(url_result["legitimate_probability"], 4),
                "confidence": round(url_result["confidence"], 4),
            },
            "metadata": {
                "verdict": meta_result["verdict"],
                "phishing_probability": round(meta_result["phishing_probability"], 4),
                "legitimate_probability": round(meta_result["legitimate_probability"], 4),
                "confidence": round(meta_result["confidence"], 4),
            },
        }
    }

@app.post("/explain")
def explain_email(req: EmailRequest):

    # 1. Decode base64
    try:
        raw_bytes = base64.b64decode(req.raw_email_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    # 2. Parse email text
    subject, body = parse_eml_text(raw_bytes)
    full_text = f"Subject: {subject}\n\n{body}".strip()

    if not full_text:
        raise HTTPException(status_code=422, detail="Could not extract text from email")

    # 3. Run LIME
    try:
        exp = agents["lime"].explain_instance(
            full_text,
            predict_proba_for_lime,
            num_features=10,
            num_samples=400,
            labels=[1],
        )
        lime_features = [
            {
                "token": token,
                "weight": round(float(weight), 4),
                "direction": "phishing" if weight > 0 else "safe"
            }
            for token, weight in exp.as_list(label=1)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME explanation failed: {str(e)}")

    return {
        "lime_features": lime_features,
        "num_features": len(lime_features)
    }