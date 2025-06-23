import json
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Union
from io import StringIO

# Import LangChain specific message types if available.
# This ensures graceful handling even if LangChain itself isn't fully installed.
try:
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    logging.warning("LangChain message types not found. Chat history will be processed as dictionaries only.")
    LANGCHAIN_MESSAGES_AVAILABLE = False
    # Define dummy classes to prevent NameErrors if not imported
    class HumanMessage:
        def __init__(self, content): self.content = content
    class AIMessage:
        def __init__(self, content): self.content = content


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import google.generativeai as genai
except ImportError:
    logging.error("Google Generative AI SDK is not installed. Please run 'pip install google-generativeai' to use the GeminiAgent.")
    pass

class GeminiAgent:
    """
    Uses Gemini 1.5 Flash model via Google Generative AI SDK for efficient language understanding.
    """

    def __init__(self, api_key: str):
        """
        Initializes the GeminiAgent with a Google API key.

        Args:
            api_key (str): The Google API key required to authenticate with Gemini.
        """
        if not api_key:
            raise ValueError("Google API key is required to initialize GeminiAgent.")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logging.info("Gemini 1.5 Flash model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model with provided API key: {e}")
            raise ConnectionError(f"Could not connect to Gemini API. Check your API key and network connection: {e}")

    def generate_response(self, data: Union[pd.DataFrame, str, None], query: str, context: Any = None, chat_history: List[Any] = None) -> str: # chat_history can now contain LangChain messages
        """
        Builds a conversational prompt with Chain-of-Thought instructions and uses Gemini
        to generate a response based on data, query, and context.

        Args:
            data (Union[pd.DataFrame, str, None]): The data (DataFrame or string) being analyzed, can be None.
            query (str): The user's latest query.
            context (Any, optional): Additional context (e.g., analytics insights, retrieved documents). Defaults to None.
            chat_history (List[Any], optional): A list of previous chat messages (can be dicts or LangChain messages). Defaults to None.

        Returns:
            str: The generated response from the Gemini model.
        """
        prompt = self._build_prompt(data, query, context, chat_history)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3, # Controls randomness. Lower for more deterministic, higher for more creative.
                    "top_p": 0.95,      # Nucleus sampling: filters out low probability tokens.
                    "max_output_tokens": 800 # Increased for potentially longer explanations/responses
                }
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating content from Gemini: {e}", exc_info=True)
            return f"I apologize, but I encountered an error while processing your request: {e}. Please try again."

    def _build_prompt(self, data: Union[pd.DataFrame, str, None], query: str, context: Any = None, chat_history: List[Any] = None) -> str: # chat_history can now contain LangChain messages
        """
        Constructs a comprehensive prompt for the Gemini model, incorporating
        system instructions, chat history, data context, and the user's latest query.

        Args:
            data (Union[pd.DataFrame, str, None]): The data (DataFrame or string) being analyzed.
            query (str): The user's latest question.
            context (Any, optional): Additional context (e.g., analytics insights, retrieved documents). Defaults to None.
            chat_history (List[Any], optional): A list of previous chat messages (can be dicts or LangChain messages). Defaults to None.

        Returns:
            str: The complete prompt string for the Gemini model.
        """
        system_prompt = (
            "You are a highly intelligent data analysis and visualization assistant. "
            "Your primary goal is to provide insightful and accurate answers to user questions "
            "based on the provided data, context, and conversation history. "
            "Follow these steps meticulously to formulate your response:\n"
            "1. **Analyze the User's Latest Query:** Understand the core intent and specific information requested.\n"
            "2. **Review Conversation History:** Recall previous turns to maintain context and continuity in the conversation. "
            "   Prioritize the most recent messages for direct relevance.\n"
            "3. **Examine Retrieved Context:** Carefully read any provided 'Retrieved context' or 'Data Summary/Insights'. "
            "   This information is crucial for grounding your response in factual data.\n"
            "4. **Consult Data Structure (if DataFrame):** If the data is a DataFrame, consider its columns, "
            "   data types, and overall structure to inform your analysis and avoid invalid assumptions.\n"
            "5. **Synthesize and Respond Clearly:** Combine all relevant information from the query, history, context, and data. "
            "   Provide a direct, concise, and helpful answer. If a chart was requested or generated, briefly explain its key takeaway."
            "   If the question is about the data, focus on numerical facts, trends, or observations."
            "   If it's about a document, summarize relevant sections."
            "   Always aim to be informative and accurate."
        )

        prompt_parts = [system_prompt]

        # Add data overview based on its type
        data_overview = ""
        if isinstance(data, pd.DataFrame):
            if not data.empty:
                data_overview = "Current Data Schema (DataFrame):\n"
                
                # Add try-except for robustness around data.info()
                try:
                    sio = StringIO()
                    data.info(buf=sio)
                    data_overview += sio.getvalue() 
                except Exception as e:
                    logging.error(f"Error capturing DataFrame info: {e}", exc_info=True)
                    data_overview += "Error: Could not retrieve detailed DataFrame info.\n"
                
                data_overview += "\n\nSample of Data (first 5 rows):\n"
                data_overview += data.head().to_markdown(index=False)
                data_overview += f"\n\nDataset has {data.shape[0]} rows and {data.shape[1]} columns."
                # Also add summary statistics if available and meaningful (might be large)
                try:
                    numeric_cols = data.select_dtypes(include=np.number)
                    if not numeric_cols.empty:
                        data_overview += "\n\nNumeric Column Statistics (mean, std, min, max):\n"
                        data_overview += numeric_cols.describe().loc[['mean', 'std', 'min', 'max']].to_markdown()
                except Exception as e:
                    logging.warning(f"Could not generate numeric stats for prompt: {e}")
            else:
                data_overview = "Data provided is an empty DataFrame."
            
        elif isinstance(data, str):
            if data.strip():
                # For text data, provide length and first/last few lines
                data_overview = f"Current Document (text data):\n"
                data_overview += f"Length: {len(data)} characters. First 200 chars: '{data[:200]}...'\n"
                data_overview += f"Last 200 chars: '...{data[-200:]}'"
            else:
                data_overview = "No text document content provided."
        elif data is None: # Explicitly handle None data
            data_overview = "No data loaded for analysis."
        else:
            data_overview = f"Unsupported data type provided for prompt building: {type(data)}."
        
        if data_overview:
            prompt_parts.append(f"\n--- Data Overview ---\n{data_overview}\n")

        # Add chat history for conversational context
        if chat_history:
            prompt_parts.append("\n--- Conversation History ---")
            for message in chat_history[-6:]: # Include more turns for better context
                role = "unknown"
                content = "No content"

                if LANGCHAIN_MESSAGES_AVAILABLE:
                    if isinstance(message, HumanMessage):
                        role = "user"
                        content = message.content
                    elif isinstance(message, AIMessage):
                        role = "assistant"
                        content = message.content
                
                # Fallback for dictionary-based chat history, or if LangChain messages not available
                if isinstance(message, dict):
                    role = message.get('role', 'unknown')
                    content = message.get('content', 'No content')

                if content: # Only add if content is not empty
                    prompt_parts.append(f"{role.capitalize()}: {content.strip()}")
            prompt_parts.append("--- End Conversation History ---\n")

        # Add retrieved context
        if context:
            prompt_parts.append("\n--- Retrieved Context ---\n")
            if isinstance(context, dict):
                # Ensure dictionary context is pretty-printed JSON
                prompt_parts.append(json.dumps(context, indent=2))
            elif isinstance(context, list):
                # Join list of context chunks
                for i, chunk in enumerate(context):
                    prompt_parts.append(f"Excerpt {i+1}:\n{chunk}")
            else:
                # Fallback for other context types
                prompt_parts.append(str(context))
            prompt_parts.append("--- End Retrieved Context ---\n")

        # Add the user's latest question
        final_user_prompt = f"User's latest question: {query}"
        prompt_parts.append(f"\n{final_user_prompt}")
        prompt_parts.append("\nAssistant:")

        return "\n".join(prompt_parts)
