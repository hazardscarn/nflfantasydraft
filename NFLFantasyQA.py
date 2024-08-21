import os
from dotenv import load_dotenv
import yaml
from typing import Dict, TypedDict, List, Optional
from langchain_community.vectorstores import TiDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import json
import atexit
import logging
import re
from sqlalchemy.exc import OperationalError

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    question: str
    contexts: List[Dict]
    final_answer: str

class NFLFantasyQA:
    def __init__(self, config_path='static/config.yml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.tidb_connection_string = os.getenv('TIDB_CONNECTION_URL')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        self.article_retriever = None
        self.player_retriever = None
        self.search_tool = None
        self.player_classifier = None
        self.llm = None
        
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        self.agent = self.create_nfl_fantasy_agent()
        
        # Register cleanup method
        atexit.register(self.cleanup)
    
    def create_retriever(self, table_name: str, k: int = 5, search_kwargs: Optional[Dict] = None) -> TiDBVectorStore:
        self.logger.info(f"Creating retriever for table: {table_name}")
        embeddings = GoogleGenerativeAIEmbeddings(model=self.config['EMBEDDING_MODEL'], google_api_key=self.google_api_key)
        vector_store = TiDBVectorStore.from_existing_vector_table(
            embedding=embeddings,
            connection_string=self.tidb_connection_string,
            table_name=table_name
        )
        if search_kwargs is None:
            search_kwargs = {}
        search_kwargs["k"] = k
        return vector_store.as_retriever(search_kwargs=search_kwargs)

    def create_player_classifier(self):
        self.logger.info("Creating player classifier")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-001",
            temperature=0.1,
            top_p=0.95,
            google_api_key=self.google_api_key,
            generation_config={"response_mime_type": "application/json"}
        )
        prompt = ChatPromptTemplate.from_template(
            """You are an NFL Fantasy Football expert. Determine if the following question requires information about specific players.
            
            Question: {question}
            
            Respond with JSON using the following schema:
            {{"needs_player_info": boolean}}
            
            Your response:
            """
        )
        return prompt | llm

    # def get_article_context(self, state: AgentState) -> AgentState:
    #     self.logger.info("Getting article context")
    #     docs = self.article_retriever.invoke(state["question"])
    #     state["contexts"].extend([{"source": "article", "content": doc.page_content} for doc in docs])
    #     self.logger.debug(f"Retrieved {len(docs)} article(s)")
    #     return state

    def get_article_context(self, state: AgentState) -> AgentState:
        self.logger.info("Getting article context")
        try:
            docs = self.article_retriever.invoke(state["question"])
            state["contexts"].extend([{"source": "article", "content": doc.page_content} for doc in docs])
            self.logger.debug(f"Retrieved {len(docs)} article(s)")
        except OperationalError as e:
            self.logger.error(f"Database connection error: {str(e)}")
            state["contexts"].append({
                "source": "article",
                "content": "Unable to retrieve article context due to a database connection issue."
            })
        except Exception as e:
            self.logger.error(f"Unexpected error in get_article_context: {str(e)}")
            state["contexts"].append({
                "source": "article",
                "content": "An unexpected error occurred while retrieving article context."
            })
        return state

    def get_player_context(self, state: AgentState) -> AgentState:
        self.logger.info("Checking if player context is needed")
        try:
            response = self.player_classifier.invoke({"question": state["question"]})
            self.logger.debug(f"Raw response: {response}")

            needs_player_info = False
            if isinstance(response, dict):
                needs_player_info = response.get("needs_player_info", False)
            elif hasattr(response, 'content'):
                json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
                if json_match:
                    try:
                        response_json = json.loads(json_match.group(1))
                        needs_player_info = response_json.get("needs_player_info", False)
                    except json.JSONDecodeError:
                        self.logger.error("Failed to parse JSON from response")
                else:
                    self.logger.warning("No JSON found in the response")
            else:
                self.logger.warning("Unexpected response format")

            self.logger.info(f"Needs player info: {needs_player_info}")

            if needs_player_info:
                self.logger.info("Getting player context")
                try:
                    docs = self.player_retriever.invoke(state["question"])
                    state["contexts"].extend([{"source": "player", "content": doc.page_content} for doc in docs])
                    self.logger.debug(f"Retrieved {len(docs)} player document(s)")
                except OperationalError as e:
                    self.logger.error(f"Database connection error: {str(e)}")
                    state["contexts"].append({
                        "source": "player",
                        "content": "Unable to retrieve player context due to a database connection issue."
                    })
                except Exception as e:
                    self.logger.error(f"Unexpected error in get_player_context: {str(e)}")
                    state["contexts"].append({
                        "source": "player",
                        "content": "An unexpected error occurred while retrieving player context."
                    })
        except Exception as e:
            self.logger.error(f"Error in player classification: {str(e)}")
            state["contexts"].append({
                "source": "player",
                "content": "Unable to determine if player context is needed due to an error."
            })
        return state

    def search_web(self, state: AgentState) -> AgentState:
        self.logger.info("Searching the web")
        try:
            search_results = self.search_tool.run(state["question"])
            state["contexts"].append({"source": "web", "content": search_results})
            self.logger.debug("Web search completed")
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            state["contexts"].append({
                "source": "web",
                "content": "Web search failed. Using only provided context for answering."
            })
        return state

    def generate_answer(self, state: AgentState) -> AgentState:
        self.logger.info("Generating final answer")
        article_context = "\n".join([f"Content: {ctx['content']}" for ctx in state["contexts"] if ctx['source'] == 'article'])
        player_context = "\n".join([f"Content: {ctx['content']}" for ctx in state["contexts"] if ctx['source'] == 'player'])
        web_context = "\n".join([f"Content: {ctx['content']}" for ctx in state["contexts"] if ctx['source'] == 'web'])
        
        self.logger.debug(f"Article context length: {len(article_context)}")
        self.logger.debug(f"Player context length: {len(player_context)}")
        self.logger.debug(f"Web context length: {len(web_context)}")

        prompt_template = """You are an NFL Fantasy Football expert. Use the following pieces of context to answer the user's question.

        Article Context: This context provides general information about NFL teams, players, and fantasy football strategies from various articles.
        {article_context}

        Player Context: This context provides specific information about individual NFL players, including their stats, team, position, and fantasy relevance.
        {player_context}

        Web Search Context: This context provides additional, potentially more recent information from web searches about NFL players, teams, and fantasy football trends.
        {web_context}

        Question: {question}

        If the user asks about a specific player, provide a detailed report on the player as long as it is from the given contexts, including:
        1. Current team and position
        2. Key stats from the previous season
        3. Fantasy football outlook and ranking
        4. Any recent news or developments (injuries, team changes, etc.)
        5. Comparison to other players in the same position

        If you don't have enough information to answer comprehensively, state what information is missing. Do Not make up answers.

        If any of the contexts indicate a failure in retrieval (e.g., database connection issues), mention this in your response and explain that the answer might be limited or less current due to these technical difficulties.

        **The output should be created in a markdown format.** Even though the content is in markdown format, use numbered lists where appropriate.
        At the end of your response, always include a list of references used to create the answer, formatted as:

        References:
        1. [Title 1](URL 1)
        2. [Title 2](URL 2)
        ...

        If no specific references were used, include a note stating that the information is based on general knowledge and the provided context.
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["article_context", "player_context", "web_context", "question"]
        )

        chain = PROMPT | self.llm

        response = chain.invoke({
            "article_context": article_context,
            "player_context": player_context,
            "web_context": web_context,
            "question": state["question"]
        })

        # Ensure the response is in markdown format
        if isinstance(response, str):
            markdown_response = response
        elif hasattr(response, 'content'):
            markdown_response = response.content
        else:
            markdown_response = str(response)

        # Clean up and standardize the references section
        parts = markdown_response.split("## References")
        content = parts[0].strip()
        
        if len(parts) > 1:
            references = parts[1].strip()
            # Remove any default messages from the references
            references = references.replace("This information is based on general knowledge and the provided context.", "").strip()
            references = references.replace("Information based on general knowledge and provided context.", "").strip()
            
            if references:
                markdown_response = f"{content}\n\n## References\n{references}"
            else:
                markdown_response = f"{content}\n\n## References\nNo specific sources cited. Information synthesized from provided context and expert knowledge."
        else:
            markdown_response = f"{content}\n\n## References\nNo specific sources cited. Information synthesized from provided context and expert knowledge."

        state["final_answer"] = markdown_response
        return state

    def create_nfl_fantasy_agent(self):
        self.logger.info("Creating NFL Fantasy Football Agent")
        self.article_retriever = self.create_retriever(self.config['vectordb']['article'], k=5)
        self.player_retriever = self.create_retriever(self.config['vectordb']['playerreport'], k=2)
        self.search_tool = DuckDuckGoSearchRun()
        self.player_classifier = self.create_player_classifier()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-001",
            temperature=0.3,
            top_p=0.95,
            google_api_key=self.google_api_key
        )

        workflow = StateGraph(AgentState)

        workflow.add_node("get_article_context", self.get_article_context)
        workflow.add_node("get_player_context", self.get_player_context)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("get_article_context")
        workflow.add_edge("get_article_context", "get_player_context")
        workflow.add_edge("get_player_context", "search_web")
        workflow.add_edge("search_web", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def get_answer(self, question: str) -> str:
        self.logger.info(f"Received question: {question}")
        initial_state = AgentState(question=question, contexts=[], final_answer="")
        result = self.agent.invoke(initial_state)
        self.logger.info("Answer generated")
        if isinstance(result["final_answer"], str):
            return result["final_answer"]
        elif hasattr(result["final_answer"], 'content'):
            return result["final_answer"].content
        else:
            return str(result["final_answer"])

    def cleanup(self):
        self.logger.info("Cleaning up resources...")
        for retriever_name in ['article_retriever', 'player_retriever']:
            retriever = getattr(self, retriever_name, None)
            if retriever is not None:
                try:
                    if hasattr(retriever, 'vectorstore'):
                        vs = retriever.vectorstore
                        if hasattr(vs, 'client'):
                            vs.client.close()
                            self.logger.info(f"Closed client for {retriever_name}")
                        elif hasattr(vs, 'connection'):
                            vs.connection.close()
                            self.logger.info(f"Closed connection for {retriever_name}")
                        elif hasattr(vs, 'pool'):
                            vs.pool.close()
                            self.logger.info(f"Closed pool for {retriever_name}")
                        else:
                            self.logger.warning(f"No known cleanup method for {retriever_name}")
                    else:
                        self.logger.warning(f"{retriever_name} has no vectorstore attribute")
                except Exception as e:
                    self.logger.error(f"Error during cleanup of {retriever_name}: {str(e)}")

        self.logger.info("Cleanup completed")

# Usage example (commented out)
# if __name__ == "__main__":
#     qa_system = NFLFantasyQA()
#     question = "Who is the best quarterback for fantasy football this season?"
#     answer = qa_system.get_answer(question)
#     print(f"Q: {question}\nA: {answer}")