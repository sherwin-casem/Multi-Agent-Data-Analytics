import pandas as pd
import uuid
import json
import logging
from typing import Dict, Any, List, Union, Tuple
import os # For os.makedirs and os.path.exists

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LangChain and Google Generative AI Imports ---
# These imports are now direct, assuming libraries are installed.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document # For creating LangChain Document objects
from langchain_core.messages import HumanMessage, AIMessage # For handling chat history in LangChain compatible format

# Agent modules
from agents.gemini_agent import GeminiAgent
from agents.analytics_agent import DataAnalyticsAgent
from agents.visualization_agent import VisualizationAgent


class AgentCoordinator:
    """
    Coordinates interactions between different AI agents (Data Analytics, Visualization, Gemini)
    and integrates with LangChain for RAG capabilities using Gemini models.
    """
    def __init__(self, api_key: str, chroma_db_dir: str = "chroma_db_data"):
        """
        Initializes the AgentCoordinator with various AI agents and LangChain RAG components.

        Args:
            api_key (str): The API key for Gemini and other agents.
            chroma_db_dir (str): Directory for ChromaDB persistence (used by LangChain Chroma).
        """
        if not api_key:
            raise ValueError("API key is required for initializing AgentCoordinator and its agents.")

        self.gemini_agent = GeminiAgent(api_key=api_key)
        self.analytics_agent = DataAnalyticsAgent(api_key=api_key)
        self.visualization_agent = VisualizationAgent(api_key=api_key)
        
        # Initialize LangChain components directly
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        self.chroma_db_dir = chroma_db_dir
        os.makedirs(self.chroma_db_dir, exist_ok=True)
        self.vectorstore: Chroma = None # Will be initialized dynamically when data is processed
        self.active_collection_name = None # To manage specific collection if needed

        logging.info("AgentCoordinator initialized with LangChain RAG components and direct ChromaDB integration.")

    def _process_text_for_rag(self, text_content: str, collection_prefix: str = "rag_collection") -> str:
        """
        Processes text content for RAG by splitting it into documents and
        storing them in a Chroma vector store.

        Args:
            text_content (str): The raw text content to be processed.
            collection_prefix (str): Prefix for the Chroma collection name.

        Returns:
            str: The name of the collection created/used.
        """
        # Create a unique collection name based on content hash or UUID
        collection_name = f"{collection_prefix}_{uuid.uuid5(uuid.NAMESPACE_DNS, text_content[:100]).hex}" # Simple hash for demo
        
        # Convert raw text into LangChain Documents
        documents = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text_content)]
        
        try:
            # Initialize or load Chroma vector store
            # This creates a new collection if it doesn't exist, otherwise loads it
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_dir,
                collection_name=collection_name
            )
            self.vectorstore.persist() # Persist the collection
            logging.info(f"Text processed and stored in Chroma collection: {collection_name}")
            self.active_collection_name = collection_name
            return collection_name
        except Exception as e:
            logging.error(f"Error processing text for RAG via LangChain Chroma: {e}", exc_info=True)
            self.vectorstore = None
            self.active_collection_name = None
            return None

    def _get_rag_context(self, query: str) -> List[str]:
        """
        Retrieves relevant context chunks from the active vector store using LangChain's retriever.

        Args:
            query (str): The user's query.

        Returns:
            List[str]: A list of retrieved document contents.
        """
        if not self.vectorstore:
            logging.warning("Vector store not initialized. Cannot retrieve RAG context.")
            return []
        
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
            # LangChain's retriever returns Document objects, extract page_content
            retrieved_docs = retriever.invoke(query)
            context_chunks = [doc.page_content for doc in retrieved_docs]
            logging.info(f"Retrieved {len(context_chunks)} RAG context chunks.")
            return context_chunks
        except Exception as e:
            logging.error(f"Error retrieving RAG context from vector store: {e}", exc_info=True)
            return []

    def handle_chat_query(self, data: Union[pd.DataFrame, str, None], query: str, chat_history: List[Dict]) -> Tuple[str, Any]:
        """
        Handles chat queries by orchestrating calls to relevant AI agents and LangChain RAG.

        Args:
            data (pd.DataFrame | str | None): The input data (Pandas DataFrame, string, or None).
            query (str): The user's chat query.
            chat_history (list): A list of previous chat messages for context.

        Returns:
            tuple[str, Any]: A tuple containing the AI agent's response (string) and
                             a chart object (Plotly Figure or None).
        """
        logging.info(f"Handling chat query. Data type: {type(data)}. Query: '{query}'")
        rag_context_str = ""
        fig = None
        chart_explanation = None
        response_text = "I'm sorry, I couldn't process your request. Please ensure a valid file is uploaded and try again."

        if data is None:
            error_msg = "No data provided or file processing failed. Please upload a valid file."
            logging.error(error_msg)
            error_chart_config = {'type': 'error', 'message': error_msg, 'title': 'Data Input Error'}
            # Pass a dummy empty DataFrame to create_chart for error visualization
            fig = self.visualization_agent.chart_generator.create_chart(pd.DataFrame(), error_chart_config) 
            return error_msg, fig # <-- IMMEDIATE RETURN HERE

        # Convert chat history to LangChain format for LLM context
        langchain_chat_history = [
            HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
            for msg in chat_history
        ]

        if isinstance(data, pd.DataFrame):
            logging.info("Processing structured data (DataFrame).")
            # For RAG on DataFrame, convert to string (e.g., markdown or JSON)
            if not data.empty:
                df_as_text = data.to_markdown(index=False) 
                self._process_text_for_rag(df_as_text, collection_prefix="df_collection")
                if self.vectorstore:
                    rag_context_chunks = self._get_rag_context(query)
                    rag_context_str = "\n".join(rag_context_chunks)
            else:
                logging.warning("DataFrame is empty. No RAG context will be generated from DataFrame.")

            # Perform data analytics and get response from analytics agent directly
            analytics_output = self.analytics_agent.analyze_data(data, query, rag_context_str, langchain_chat_history)
            response_text = analytics_output['response'] # Primary response from GeminiAgent via DataAnalyticsAgent

            # Generate visualization based on query
            fig, chart_explanation = self.visualization_agent.generate_chart(data, query, rag_context_str, langchain_chat_history)
            
            # Combine responses (if analytics agent doesn't fully cover)
            insights = analytics_output.get('key_insights', [])
            if insights:
                response_text += "\n\n**Key Insights:**\n- " + "\n- ".join(insights)
            quality_score = analytics_output.get('data_quality_score')
            if quality_score:
                response_text += f"\n\n**Data Quality Score:** {quality_score}"
            recommendations = analytics_output.get('recommendations', [])
            if recommendations:
                response_text += "\n\n**Recommendations:**\n- " + "\n- ".join(recommendations)
            
            if chart_explanation:
                response_text += f"\n\n**Chart Explanation:**\n{chart_explanation}"

            return response_text, fig

        elif isinstance(data, str):
            logging.info("Processing unstructured data (string) with LangChain RAG.")
            # Process text data for RAG
            if data.strip():
                self._process_text_for_rag(data, collection_prefix="doc_collection")
                if self.vectorstore:
                    rag_context_chunks = self._get_rag_context(query)
                    rag_context_str = "\n".join(rag_context_chunks)
            else:
                logging.warning("Text data is empty. No RAG context will be generated.")

            # Use LangChain's RetrievalQA chain for direct Q&A on text documents
            if self.vectorstore:
                try:
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff", # 'stuff' combines all docs into one prompt
                        retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    # For now, let's pass query directly. Context from retrieval will be used.
                    response_from_rag = qa_chain.invoke({"query": query})
                    response_text = response_from_rag.get("result", "Could not generate a response from RAG.")
                    if response_from_rag.get("source_documents"):
                        sources = "\n".join([doc.page_content[:100] + "..." for doc in response_from_rag["source_documents"]])
                        response_text += f"\n\n**Sources:**\n{sources}"
                    
                except Exception as e:
                    logging.error(f"Error in LangChain RetrievalQA for text data: {e}", exc_info=True)
                    response_text = f"An error occurred while performing RAG on the document: {e}"
            else:
                response_text = self.gemini_agent.generate_response(data, query, rag_context_str, langchain_chat_history)
                logging.warning("Vector store not ready for text RAG. Falling back to direct Gemini Agent response.")

            # No charts for unstructured text
            fig = None
            chart_explanation = "No chart generated for unstructured text data."
            
            return response_text, fig
        
        else: # Handle any other unexpected data types
            error_msg = f"Cannot process data of type {type(data)}. Expected pandas.DataFrame or str."
            logging.error(error_msg)
            error_chart_config = {'type': 'error', 'message': error_msg, 'title': 'Unsupported Data Type'}
            # Pass a dummy empty DataFrame to create_chart for error visualization
            fig = self.visualization_agent.chart_generator.create_chart(pd.DataFrame(), error_chart_config) 
            return error_msg, fig # <-- IMMEDIATE RETURN HERE


if __name__ == "__main__":
    # Example usage for direct AgentCoordinator testing
    # Ensure you replace "YOUR_GOOGLE_API_KEY" with a real key or use environment variables.
    # import os
    # api_key = os.getenv("GOOGLE_API_KEY") # Load from environment variable for security

    # if api_key:
    #     coordinator = AgentCoordinator(api_key=api_key)

    #     # Example with DataFrame (ensure sample_data.csv exists)
    #     # try:
    #     #     sample_df = pd.read_csv("sample_data.csv") 
    #     #     response, chart = coordinator.handle_chat_query(sample_df, "Summarize the data and show me a chart of column A vs B.", [])
    #     #     print("\n--- Coordinator Response (DataFrame) ---")
    #     #     print(response)
    #     #     if chart:
    #     #         print("Chart object generated. In a Streamlit app, this would be displayed.")
    #     # except FileNotFoundError:
    #     #     logging.error("sample_data.csv not found. Please create it for DataFrame example.")
    #     # except Exception as e:
    #     #     logging.error(f"Error in DataFrame example: {e}", exc_info=True)

    #     # Example with text data
    #     # text_data = "The quick brown fox jumps over the lazy dog. This is a test sentence about natural language processing. LangChain helps build LLM applications easily."
    #     # response, chart = coordinator.handle_chat_query(text_data, "What is LangChain?", [])
    #     # print("\n--- Coordinator Response (Text) ---")
    #     # print(response)
    # else:
    #     logging.warning("API key not found. Please set GOOGLE_API_KEY environment variable for testing.")
    pass