import os

from agents.linkedin_lookup_agent import lookup
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    _ = load_dotenv()

    linkedin_profile_url = lookup("Nelmin Jay M. Anoc")

    information = scrape_linkedin_profile(linkedin_profile_url)

    summary_template = """
        given the linkedin information {information} about a person from I
        want you to create:
        1. a short summary
        2. two interesting facts about them

        your tone should be like rick from rick and morty 
    """

    summary_prompt_template = ChatPromptTemplate.from_template(summary_template)

    llm = ChatOpenAI(
        temperature=0, model="gpt-4o", api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke({"information": information})

    print(res)
