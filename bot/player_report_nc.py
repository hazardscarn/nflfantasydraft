import csv
import os
from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools import Tool
import logging
import sys
import re
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the state structure
class AgentState(TypedDict):
    messages: List[BaseMessage]
    player_name: str
    player_info: Dict[str, str]
    status: str

# Initialize tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding up-to-date information about NFL players"
    )
]

# Initialize Ollama LLM
llm = Ollama(model="llama3.1:8b")

# Define prompts
PLAYER_INFO_PROMPT = ChatPromptTemplate.from_template("""
You are an NFL fantasy football expert tasked with gathering the latest information about a given player.
For each aspect listed below, you MUST use the Search tool to find the most recent and accurate information.
Do not rely on your pre-existing knowledge. Always search for the latest data.

Focus on these aspects, using a separate search for each:
1. Age, weight, height, current position, and team
2. College background and draft information
3. NFL experience and career highlights
4. 2024 fantasy football outlook and projections
5. Last season's performance stats (or college stats and draft position for rookies)
6. Current contract status and any recent contract news
7. Recent news or developments (injuries, team changes, coaching impacts, etc.)
8. Comparison to other current players in the same position (use stats to support)
9. Identify 5 current NFL players most similar in fantasy football value and playing style

For each aspect, follow this process:
1. Formulate a specific search query
2. Use the Search tool with that query
3. Analyze the search results
4. Summarize the relevant information

Use the following format:

Question: the specific aspect you're researching
Thought: your reasoning for the search query
Action: Search
Action Input: your carefully formulated search query
Observation: the search results
Thought: your analysis of the search results
Final Answer: a concise summary of the relevant information for that aspect

Repeat this process for each of the 9 aspects listed above.

You have access to the following tools:

{tools}

Use the following tool names: {tool_names}

Begin your research on {input}:
{agent_scratchpad}
""")

REPORT_GENERATION_PROMPT = ChatPromptTemplate.from_template("""
Based on the collected information, generate a detailed player report for {player_name}.
The report should be well-structured, easy to read, and cover all the aspects researched.
Make sure to highlight key points that would be valuable for fantasy football managers.
Do not lose any information from the player's profile.

Player Information:
{player_info}

Generate a concise outlook summary (2-3 sentences) for the player's 2024 fantasy football prospects,
focusing on their projected performance, any risks or upside potential, and how they compare to other players in their position.
""")

# Create an agent
player_info_agent = create_react_agent(
    llm,
    tools,
    PLAYER_INFO_PROMPT,
    output_parser=ReActJsonSingleInputOutputParser(),
)
player_info_executor = AgentExecutor(
    agent=player_info_agent,
    tools=tools,
    handle_parsing_errors=True,
    max_iterations=10,  # Increase max iterations to allow for multiple searches
)

def get_player_report(player_name: str) -> str:
    logging.info(f"{player_name}: Gathering information...")
    try:
        result = player_info_executor.invoke({
            "input": player_name,
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools]),
            "agent_scratchpad": ""
        })
        
        logging.info(f"{player_name}: Generating report...")
        report_chain = REPORT_GENERATION_PROMPT | llm
        report = report_chain.invoke({"player_name": player_name, "player_info": result['output']})
        
        logging.info(f"{player_name}: Completed")
        return report
    except Exception as e:
        logging.error(f"Error processing {player_name}: {str(e)}")
        return f"Error processing player: {str(e)}"

def clean_player_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', name)

def process_player(row: Dict[str, str], output_folder: str) -> None:
    player_name = row['player']
    try:
        outlook = get_player_report(player_name)
        
        clean_name = clean_player_name(player_name)
        output_file = os.path.join(output_folder, f"{clean_name}.csv")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = list(row.keys()) + ['Outlook']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            row['Outlook'] = outlook
            writer.writerow(row)
        
        logging.info(f"Report saved for {player_name}")
    except Exception as e:
        logging.error(f"Error saving report for {player_name}: {str(e)}")

def process_players(input_file: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            process_player(row, output_folder)
            logging.info(f"Waiting 10 minutes before processing the next player...")
            time.sleep(300)

if __name__ == "__main__":
    input_file = "bot/documents/player_list.csv"
    output_folder = "bot/documents/player_outlook"
    process_players(input_file, output_folder)
    logging.info("All players processed. Check the output folder for results.")