from pydantic import BaseModel

class TicketAnalysis(BaseModel):
    """Result of ticket analysis"""

    category: str
    subcategory: str
    priority: str
    assigned_team: str
    technical_complexity: str
    required_expertise: list[str]
    escalation_required: bool
    reasoning: str
    confidence_score: float
    immediate_actions: list[str]
    similar_tickets: list[dict]