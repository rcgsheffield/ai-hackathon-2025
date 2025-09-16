# %%
from multiprocessing import context
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from topdesk_analyser.agent import get_agent
from typing import Optional

from dotenv import load_dotenv

load_dotenv(".env")

prompt = """You are an expert IT Support Analyst with 15+ years of experience in enterprise technical support. Your expertise includes:

- Incident categorization and priority assessment
- Technical complexity evaluation 
- Team routing and workload distribution
- Escalation procedures and SLA management
- Risk assessment and business impact analysis

CORE RESPONSIBILITIES:
1. Analyze support tickets using historical patterns and technical knowledge
2. Determine appropriate categorization based on issue type and technical requirements
3. Assess priority considering business impact, user count, and urgency
4. Route tickets to teams based on technical expertise and workload capacity
5. Identify escalation needs and potential risks

AVAILABLE SUPPORT TEAMS & EXPERTISE:
- Level 1 Support: Basic troubleshooting, password resets, general user guidance, simple software issues
- Applications Support: Software installation/configuration, desktop applications, user training, RStudio, Python, R, Office suite
- Infrastructure Team: Network issues, server problems, storage systems, backup/recovery, performance issues, hardware
- HPC Team: High-performance computing, cluster management, job scheduling (SLURM/PBS), scientific computing, GPU workloads
- Security Team: Security incidents, access control, compliance, suspicious activity, data breaches
- Data Services: Database issues, data migration, ETL processes, data governance, BI tools

PRIORITY GUIDELINES:
- P1 (Critical): System outages, security breaches, data loss, complete service failures affecting multiple users/departments
- P2 (High): Significant functionality loss, performance issues affecting productivity, partial outages, approaching deadlines
- P3 (Medium): Moderate issues affecting individual users or small teams, software bugs with workarounds
- P4 (Low): Minor issues, enhancement requests, training requests, documentation updates

COMPLEXITY ASSESSMENT:
- Critical: Multi-system failures, security incidents, data integrity issues requiring senior expertise
- High: Complex configurations, system integrations, performance optimization, specialized technical knowledge needed
- Medium: Standard software issues, configuration changes, troubleshooting requiring technical skills
- Low: Basic user issues, simple configurations, routine maintenance tasks

Always base your analysis on:
1. Historical similar incidents and their outcomes
2. Technical complexity and required expertise
3. Business impact and user count affected
4. Urgency and timeline constraints
5. Available team capacity and expertise

Provide thorough reasoning for all decisions and maintain consistency with historical patterns while adapting to unique circumstances."""

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

agent = get_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    prompt=prompt,
    tools=[],
    response_format=TicketAnalysisSchema,
)
# %%
from vector_db import TopDeskVectorDB
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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



def make_user_prompt(context: str) -> str:
    """Create a detailed user prompt for ticket analysis"""

    prompt = f"""{context}

    === ANALYSIS REQUEST ===
    Based on your expertise and the historical patterns above, provide a comprehensive analysis of this support ticket. Consider:

    1. What type of issue is this and how does it compare to similar historical cases?
    2. What is the appropriate category and subcategory based on the technical nature?
    3. What priority should this receive considering business impact, urgency, and affected users?
    4. Which team has the right expertise and capacity to handle this effectively?
    5. What is the technical complexity and what specific skills are needed?
    6. Does this require immediate escalation or special attention?
    7. What immediate actions should be taken to begin resolution?

    Use your expertise to make intelligent decisions. Reference the similar tickets where relevant, but adapt to the unique aspects of this new issue. Be thorough in your reasoning and confident in your recommendations."""
    return prompt

def get_similar_tickets(query: str, n_results: int = 8, collection_name='topdesk_incidents') -> list[dict]:
    """Get similar tickets from ChromaDB with rich context"""
    
    client = TopDeskVectorDB().get_langchain_client()
    embedding_model = "all-MiniLM-L6-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory='./topdesk_vectordb',  # Where to save data locally
    )
    incidents = vectorstore.similarity_search(query, k=n_results)
    
    return incidents

similar_tickets = get_similar_tickets("Email not working", n_results=5)
# %%

def format_similar_tickets(tickets):
    formatted_tickets = []
    for ticket in tickets:
        ticket_data = ticket.page_content
        metadata = ticket.metadata
        string = f'Ticket Metadata: {metadata}\nTicket Data: {ticket_data}'
        formatted_tickets.append(string)
    return '\n\n'.join(formatted_tickets)

formatted_tickets = format_similar_tickets(similar_tickets)

print(formatted_tickets)

user_problem = "Email not working"
# %%
from langchain_core.messages import SystemMessage

def pipeline(state):
    user_problem = state['messages'][-1].content
    similar_tickets = get_similar_tickets(user_problem, n_results=5)
    formatted_tickets = format_similar_tickets(similar_tickets)
    system_prompt = SystemMessage(
        content=prompt + "\n\n" + make_user_prompt(formatted_tickets)
    )
    return [system_prompt, *list(state['messages'])]

rag_agent = create_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[],
    prompt=pipeline,
    response_format=TicketAnalysisSchema,)

response = rag_agent.invoke({'messages': [{'role': 'user', 'content': user_problem}]})


# %%
def format_ticket_analysis(analysis: dict) -> str:
    return (
        f"Category: {analysis.get('category')}\n"
        f"Subcategory: {analysis.get('subcategory')}\n"
        f"Priority: {analysis.get('priority')}\n"
        f"Assigned Team: {analysis.get('assigned_team')}\n"
        f"Technical Complexity: {analysis.get('technical_complexity')}\n"
        f"Required Expertise: {', '.join(analysis.get('required_expertise', []))}\n"
        f"Escalation Required: {analysis.get('escalation_required')}\n"
        f"Reasoning: {analysis.get('reasoning')}\n"
        f"Immediate Actions: {', '.join(analysis.get('immediate_actions', []))}\n"
        f"Estimated Resolution Time: {analysis.get('estimated_resolution_time')}\n"
        f"Business Impact: {analysis.get('business_impact')}\n"
        f"Risk Factors: {', '.join(analysis.get('risk_factors', []))}"
    )

print(format_ticket_analysis(response['structured_response'].model_dump()))
# %%
