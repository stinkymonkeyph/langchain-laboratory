import os
from typing import Tuple

from agents.linkedin_lookup_agent import lookup
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from third_parties.linkedin import scrape_linkedin_profile
from output_parser import Summary, summary_parser


def ice_break_with(name: str) -> Tuple[Summary, str | None]:
    _ = load_dotenv()
    linkedin_profile_url = lookup(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    summary_template = """
        given the linkedin information {information} about a person from I
        want you to create:
        1. a short summary
        2. two interesting facts about them
        3. interesting role or project

        your tone should be like rick from rick and morty 
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(
        temperature=0, model="gpt-4o", api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )

    chain = summary_prompt_template | llm | summary_parser

    res = chain.invoke({"information": linkedin_data})
    return res, linkedin_data.get("profile_pic_url")


if __name__ == "__main__":
    res, profile_pic_url = ice_break_with("Nelmin Jay M. Anoc")
    print(f"summary: {res.summary}")
    print(f"facts: {res.facts}")
