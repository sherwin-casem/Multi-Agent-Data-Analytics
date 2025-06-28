# agents/analytics_agent.py

import pandas as pd
import json
import logging
from typing import Dict, Any, List, Union, Optional
from agents.gemini_agent import GeminiAgent

class DataAnalyticsAgent:
    """
    Analyzes data and uses the Gemini agent with RAG context to generate insights.
    """
    def __init__(self, api_key: str):
        self.gemini_agent = GeminiAgent(api_key=api_key)

    def analyze_data(
        self,
        data: Union[pd.DataFrame, str],
        query: str = "Summarize the dataset",
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        response = ""
        key_insights = []
        data_quality_score = {}
        recommendations = []

        if isinstance(data, pd.DataFrame):
            logging.info("DataAnalyticsAgent processing DataFrame.")

            query_result_df_or_str = self._execute_dataframe_query(data, query)

            if isinstance(query_result_df_or_str, pd.DataFrame):
                processed_data_summary = self._summarize_dataframe(query_result_df_or_str)

                response_table = "Here are the top results from your query:\n\n"
                markdown_result = query_result_df_or_str.to_markdown(index=False) or ""
                response_table += markdown_result
                response_table += "\n\n"

                response_from_gemini = self.gemini_agent.generate_response(
                    data=None,
                    query=query,
                    context=f"{response_table}Original Data Summary:\n{json.dumps(processed_data_summary, indent=2)}",
                    chat_history=chat_history,
                )
                response = response_table + response_from_gemini

                key_insights = self._extract_key_insights(data)
                data_quality_score = self._calculate_data_quality(data)
                recommendations = self._generate_recommendations(data_quality_score)

            else:
                response += str(query_result_df_or_str) + "\n\n"
                response_from_gemini = self.gemini_agent.generate_response(
                    data=None,
                    query=query,
                    context=context or "",
                    chat_history=chat_history,
                )
                response += response_from_gemini

                key_insights = self._extract_key_insights(data)
                data_quality_score = self._calculate_data_quality(data)
                recommendations = self._generate_recommendations(data_quality_score)

        elif isinstance(data, str):
            logging.info("DataAnalyticsAgent processing unstructured text.")
            response = self.gemini_agent.generate_response(
                data=None,
                query=query,
                context=context or "",
                chat_history=chat_history,
            )

        else:
            logging.warning(f"Unsupported data type for DataAnalyticsAgent: {type(data)}")
            response = "I cannot perform analysis on the provided data type."

        return {
            "response": response,
            "key_insights": key_insights,
            "data_quality_score": data_quality_score,
            "recommendations": recommendations,
        }

    # --- Placeholder Methods ---

    def _execute_dataframe_query(self, df: pd.DataFrame, query: str) -> Union[pd.DataFrame, str]:
        # Implement logic if needed, else return original df
        return df

    def _summarize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict()
        }

    def _extract_key_insights(self, df: pd.DataFrame) -> List[str]:
        return [f"Most common value in {col}: {df[col].mode()[0]}" for col in df.select_dtypes(include="object").columns]

    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "completeness": 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            "unique_ratio": df.nunique().mean() / df.shape[0],
        }

    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        recs = []
        if quality_metrics.get("completeness", 1.0) < 0.9:
            recs.append("Address missing values to improve data completeness.")
        if quality_metrics.get("unique_ratio", 1.0) < 0.2:
            recs.append("Some columns may lack variability. Consider enrichment.")
        return recs
