import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from tools.tools import get_profile_url_travily

load_dotenv()


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0, model="gpt-4o", api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )

    template = """
        given the full name {name_of_person} I want you to get it me a link to their linkedin profile page.
        your answer should contain only a URL, and it should be a Linkedin url 
        if you are having a difficult time searching for linkedin profile,
        you can try permutating the name to get the best result
        result should be something like linkedin.com/
    """

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Search for linkedin profile",
            func=get_profile_url_travily,
            description="useful for when you need to get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url
