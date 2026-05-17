from pydantic import BaseModel
from typing import Any

class EmailRequest(BaseModel):
    raw_email_b64: str

#TODO: define fully once model team confirms output shape
class EmailResponse(BaseModel):
    verdict: str
    confidence: float
    agents: Any