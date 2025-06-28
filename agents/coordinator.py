# --- REPLACE THE ENTIRE CONTENT OF coordinator.py WITH THIS ---

from .analytics_agent import DataAnalyticsAgent
from .visualization_agent import VisualizationAgent
from .gemini_agent import GeminiAgent  # <-- NEW
from utils.vector_db_handler import VectorDBHandler
from utils.db_connector import DBConnector # <-- NEW
import pandas as pd
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# --- LangGraph State Definition (Step 8) ---
class AgentState(TypedDict):
    query: str
    data: object
    chat_history: list
    analysis_context: Annotated[list, operator.add]
    response: str

class AgentCoordinator:
    """
    Coordinates interactions between agents, using a dual RAG pipeline and a stateful graph.
    """
    def __init__(self, gemini_api_key: str):
        self.analytics_agent = DataAnalyticsAgent(api_key=gemini_api_key)
        self.visualization_agent = VisualizationAgent(api_key=gemini_api_key)
        self.gemini_agent = GeminiAgent(api_key=gemini_api_key) # <-- NEW
        self.vector_db_handler = VectorDBHandler(api_key=gemini_api_key)
        self.db_connector = DBConnector() # <-- NEW
        self.active_collection = None

        # --- LangGraph Workflow (Step 8) ---
        # Note: This is a simplified conceptual graph.
        # A full implementation would require more complex routing logic.
        workflow = StateGraph(AgentState)
        workflow.add_node("rag_pipeline", self._run_rag_pipeline)
        workflow.add_node("llm_call", self._run_llm_call)
        workflow.set_entry_point("rag_pipeline")
        workflow.add_edge("rag_pipeline", "llm_call")
        workflow.add_edge("llm_call", END)
        self.graph = workflow.compile()
        print("LangGraph workflow compiled.")


    def get_data_from_db(self, query: str):
        """Fetches data from the configured PostgreSQL database."""
        return self.db_connector.fetch_data(query)

    def get_analytics_insights(self, data):
        return self.analytics_agent.analyze_data(data)

    def generate_visualization(self, data, query):
        return self.visualization_agent.generate_chart(data, query)

    def handle_chat_query(self, data, query, chat_history: list):
        """
        Handles chat queries by invoking the LangGraph workflow.
        """
        initial_state: AgentState = {
            "query": query,
            "data": data,
            "chat_history": chat_history,
            "analysis_context": [],
            "response": ""
        }
        final_state = self.graph.invoke(initial_state)
        return final_state['response']

    def _run_rag_pipeline(self, state: AgentState) -> dict:
        """Node for the RAG pipeline to gather context."""
        query = state['query']
        data = state['data']
        context = None

        if isinstance(data, pd.DataFrame):
            # Simple RAG for structured data (can be expanded)
            try:
                numeric_cols = data.select_dtypes(include=['number']).columns
                context = {"data_summary": data[numeric_cols].describe().to_dict()}
            except Exception:
                context = {"data_summary": "Could not generate summary."}
        elif isinstance(data, str):
            # RAG for unstructured data
            if self.active_collection is None:
                self.active_collection = self.vector_db_handler.process_text(data, collection_prefix="default")
            context = self.vector_db_handler.get_context(query)

        return {"analysis_context": [context] if context else []}

    def _run_llm_call(self, state: AgentState) -> dict:
        """Node for calling the LLM with the gathered context."""
        response = self.gemini_agent.generate_response(
            data=state['data'],
            query=state['query'],
            context=state['analysis_context'],
            chat_history=state['chat_history']
        )
        return {"response": response}
