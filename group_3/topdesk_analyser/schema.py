from pydantic import BaseModel, Field
from typing import Optional

class IncidentSchema(BaseModel):
    id: str
    number: str
    category: str
    subcategory: str
    object_type: str
    object_name: Optional[str]
    research_discipline: Optional[str]
    software_required: Optional[str]
    operator_group: Optional[str]
    operator: Optional[str]
    caller_name: Optional[str]
    caller_department: Optional[str]
    location: Optional[str]
    priority: Optional[str]
    status: Optional[str]
    source: Optional[str]
    document_type: Optional[str]
    brief_description: Optional[str]
    impact: Optional[str]
    urgency: Optional[str]
    creation_date: str
    modification_date: str
    call_date: str

class TicketAnalysisSchema(BaseModel):
    category: str
    subcategory: str
    priority: str
    assigned_team: str
    technical_complexity: str
    required_expertise: list[str]
    escalation_required: bool
    reasoning: str
    confidence_score: float = Field(..., ge=0, le=1)
    immediate_actions: list[str]
    estimated_resolution_time: str
    business_impact: str
    risk_factors: list[str]