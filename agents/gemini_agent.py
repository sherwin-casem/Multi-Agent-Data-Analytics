# --- CREATE THIS NEW FILE ---

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

class GeminiAgent:
    """
    Integrates the Google Gemini Pro model for advanced language understanding and generation.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API Key is required.")
        # Use the new Flash model for speed and cost-effectiveness
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", max_tokens=512, top_k=30, top_p=0.95, temperature=0.3)

    from typing import Optional

    def generate_response(self, data, query: str, context=None, chat_history: Optional[list] = None):
        """
        Builds a conversational prompt and uses the Gemini model to generate a response.
        """
        system_prompt = """
        You are an expert football (soccer) data analyst. Your task is to analyze the provided context, data summary, and conversation history to answer the user's question.
        
        Follow these steps:
        1.  **Analyze the user's latest query:** Understand the specific football-related question.
        2.  **Review the conversation history:** Is this a follow-up question? Maintain context from previous turns.
        3.  **Examine the retrieved context:** This contains factual data (from a CSV, database, or text document) that is relevant to the query. This is your source of truth.
        4.  **Synthesize and Respond:** Formulate a clear, concise, and insightful answer based on all available information.
        
        **RULES:**
        -   ALWAYS base your answers on the provided context. Do not use outside knowledge.
        -   If the context doesn't contain the answer, state that the information is not available in the provided data.
        -   Be direct. Do not mention "based on the context" or "according to the document."
        -   Format your answers for readability (e.g., use bullet points for lists).
        """

        # Build the prompt dynamically
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # Dynamically add chat history
            *[(msg["role"], msg["content"]) for msg in (chat_history or [])[-4:]],
            # Add the context and the final user query
            ("user", "CONTEXT:\n{context}\n\nLATEST QUESTION:\n{query}")
        ])

        # Prepare context string
        context_str = "No specific context retrieved."
        if context:
            context_str = json.dumps(context, indent=2) if isinstance(context, (dict, list)) else str(context)

        chain = prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context_str,
            "query": query
        })

        return response
