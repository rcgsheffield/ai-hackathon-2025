from langchain.agents import create_agent


def get_agent(model, prompt, tools=[], response_format=None):
    agent = create_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=response_format,
    )
    return agent
