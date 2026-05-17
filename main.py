from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import EmailRequest, EmailResponse
from email import message_from_string
from email.policy import default as email_default_policy
import base64

# -------------------------------------------------------
# Email parser
# Copied directly from gradio_demo_cell.py 
# -------------------------------------------------------
def parse_eml_text(raw_bytes: bytes) -> tuple[str, str]:
    """
    Parse raw email bytes into (subject, body).
    Tries UTF-8 first, falls back to latin-1.
    Prefers text/plain, falls back to text/html.
    """
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
# App setup
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: load models here once model team delivers
    # agents["url"]      = ...
    # agents["metadata"] = ...
    # agents["text"]     = ...
    print("Server started. Models not yet loaded — awaiting model team.")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="Phishing Detector API",
    description="Multi-agent phishing detection backend.",
    version="0.1.0",
    lifespan=lifespan
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

    # 2. Parse email
    subject, body = parse_eml_text(raw_bytes)
    full_text = f"Subject: {subject}\n\n{body}".strip()

    if not full_text:
        raise HTTPException(status_code=422, detail="Could not extract text from email")

    # TODO: run agents once model team delivers
    # url_result  = agents["url"].get_prediction_with_confidence(features)
    # meta_result = agents["metadata"].get_prediction_with_confidence(features)
    # text_result = run_distilbert(full_text)
    # verdict     = fuse(url_result, meta_result, text_result)

    raise HTTPException(
        status_code=503,
        detail="Models not yet available. Backend is ready — awaiting model files."
    )