#!/usr/bin/python
import os
from anthropic import Anthropic


# read in text from file
def read_file(file_name):
    with open(file_name, "r") as f:
        text = f.read()
    return text
    

# get API key
#client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
 
# initialise anthropic client
CLAUDE_API_KEY=
client = Anthropic(api_key=CLAUDE_API_KEY)


# variables
model_name="claude-sonnet-4-20250514"
role="user"

# organise prompt
prompt_file="./input/prompt.txt"
prompt_text=read_file(prompt_file)
research_it_website_content_file="./input/research_it_website_content.txt"
research_it_website_content=read_file(research_it_website_content_file)
research_it_requirements_output_file="./input/Research_IT_Requirements_Output.csv"
research_it_requirements_output=read_file(research_it_requirements_output_file)
grant_file="./input/grant_text.txt"
grant_text=read_file(grant_file)

content=f""" 
{prompt_text} 

Research IT Website Content: {research_it_website_content} 

The Research IT output should be provided in this CSV format: {research_it_requirements_output}

Grant Text: {grant_text}
"""


# make request
response = client.messages.create(
    model=model_name,
    max_tokens=1000,
    messages=[
        {
            "role": role,
            "content": content,
        }
    ]
)

print(response.content[0].text)



