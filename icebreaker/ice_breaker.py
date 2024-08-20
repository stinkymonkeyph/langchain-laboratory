from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import os

information = """
    Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a
    businessman and investor known
    for his key roles in the space company SpaceX
    and the automotive company Tesla,
    Inc. Other involvements include ownership of X Corp.,
    the company that
    operates the social media platform X
    (formerly known as Twitter), and his role in the founding of
    The Boring Company, xAI, Neuralink, and
    OpenAI. He is one of the wealthiest individuals in the world;
    as of August 2024 Forbes estimates his net worth to be US$241 billion
"""

if __name__ == '__main__':
    load_dotenv()

    summary_template = """
        given the information {information} about a person from I
        want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = ChatPromptTemplate.from_template(
        summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o",
                     api_key=os.environ["OPENAI_API_KEY"])

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke({"information": information})

    print(res)
