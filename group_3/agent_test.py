# %%
from multiprocessing import context
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from topdesk_analyser.agent import get_agent
from typing import Optional

from topdesk_analyser.schema import TicketAnalysisSchema
from topdesk_analyser.utils import (
    format_ticket_analysis,
    pipeline,
)

from dotenv import load_dotenv

load_dotenv(".env")


# %%

user_problem = """
My name is Alexis, I need help with analysing my research data.
I'm trying to do geospatial analytics in Python. But it's just NOT working?!

Key software or programming language:
- Python
How did you hear about us?
- Faculty communications

"""


rag_agent = create_agent(
    model="ollama:gpt-oss:20b",
    tools=[],
    prompt=pipeline,)

response = rag_agent.invoke({'messages': [{'role': 'user', 'content': user_problem}]})

print(format_ticket_analysis(response['structured_response'].model_dump()))
# %%