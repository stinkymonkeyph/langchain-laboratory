from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import tool
from langchain_core.tools import Tool, render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser
)
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Union

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for loc_tool in tools:
        if loc_tool.name == tool_name:
            return loc_tool
    raise ValueError(f"Tool with name {tool_name} not found.")


def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets
    the agent continue its thought process"""

    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


if __name__ == "__main__":
    tools = [get_text_length]

    template = """
        Answer the following questions as best you can.
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0, stop=["\nObservation"])
    intermediate_steps = []

    agent = ({
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    } | prompt | llm | ReActSingleInputOutputParser())

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of the text DOG ?",
            "agent_scratchpad": intermediate_steps
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    agent = ({
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    } | prompt | llm | ReActSingleInputOutputParser())

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of the text DOG ?",
            "agent_scratchpad": intermediate_steps
        }
    )

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
