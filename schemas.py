from pydantic import BaseModel
from typing import  Any, List


class TextAgentResult(BaseModel):
    verdict: str
    phishing_probability: float
    safe_probability: float
    confidence: float

class RFAgentResult(BaseModel):
    verdict: str
    phishing_probability: float
    legitimate_probability: float
    confidence: float

class AgentsResult(BaseModel):
    text: TextAgentResult
    url: RFAgentResult
    metadata: RFAgentResult

class EmailRequest(BaseModel):
    raw_email_b64: str

class EmailResponse(BaseModel):
    verdict: str
    confidence: float
    agents: AgentsResult

class LimeFeature(BaseModel):
    token: str
    weight: float
    direction: str

class ExplainResponse(BaseModel):
    lime_features: List[LimeFeature]
    num_features: int