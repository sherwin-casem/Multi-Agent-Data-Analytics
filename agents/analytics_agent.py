import pandas as pd
import numpy as np
import logging
from typing import Union, Dict, Any, List
import re # Import regex module
import json # Import json for better context handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from agents.gemini_agent import GeminiAgent  # Ensure this file is in the same directory

class DataAnalyticsAgent:
    """
    Agent responsible for performing data analysis, extracting key insights,
    calculating data quality scores, and generating recommendations.
    It uses a GeminiAgent for generating high-level responses and a new
    method for executing basic DataFrame queries.
    """
    def __init__(self, api_key: str):
        """
        Initializes the DataAnalyticsAgent with a GeminiAgent.

        Args:
            api_key (str): The API key for the GeminiAgent.
        """
        self.gemini_agent = GeminiAgent(api_key=api_key)
        logging.info("DataAnalyticsAgent initialized.")

    def analyze_data(self, data: Union[pd.DataFrame, str], query: str = "Summarize the dataset", context: Any = None, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyzes the given data (DataFrame or string) and returns a comprehensive report.
        If the query implies data manipulation, it attempts to execute a DataFrame query.

        Args:
            data (Union[pd.DataFrame, str]): The input data to analyze.
            query (str): The user's query for analysis.
            context (Any, optional): Additional context for the GeminiAgent. Defaults to None.
            chat_history (List[Dict[str, str]], optional): Previous chat history for context. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, including a response,
                            key insights, data quality score, and recommendations.
        """
        response = ""
        key_insights = []
        data_quality_score = {}
        recommendations = []

        if isinstance(data, pd.DataFrame):
            logging.info("DataAnalyticsAgent processing DataFrame.")
            
            # Attempt to execute a DataFrame query first if applicable
            query_result_df_or_str = self._execute_dataframe_query(data, query)
            
            if isinstance(query_result_df_or_str, pd.DataFrame):
                # If a DataFrame query was executed successfully and returned a DataFrame,
                # use its summary as part of the context for Gemini.
                processed_data_summary = self.summarize(query_result_df_or_str)
                
                # Format the DataFrame result nicely for the user response
                response_table = f"Here are the top results from your query:\n\n"
                response_table += query_result_df_or_str.to_markdown(index=False)
                response_table += "\n\n" # Add extra newlines for better markdown rendering
                
                # Pass the raw DataFrame, query, and formatted table as context to GeminiAgent
                response_from_gemini = self.gemini_agent.generate_response(
                    data, # Original data still passed for overall context if needed by Gemini
                    query, 
                    context=f"{response_table}Original Data Summary:\n{json.dumps(processed_data_summary, indent=2)}", 
                    chat_history=chat_history
                )
                response = response_table + response_from_gemini
                
                # Recalculate insights and quality for the *original* data,
                # or consider if insights for the filtered data are more appropriate based on query intent.
                # For now, keeping it consistent with overall data summary.
                key_insights = self.extract_key_insights(data)
                data_quality_score = self.calculate_data_quality(data)
                recommendations = self.generate_recommendations(data_quality_score)

            else: # If _execute_dataframe_query returned a string message or didn't execute
                # Append the message from _execute_dataframe_query
                response += query_result_df_or_str + "\n\n" 
                # Then get a general response from Gemini, including existing context
                response_from_gemini = self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)
                response += response_from_gemini

                key_insights = self.extract_key_insights(data)
                data_quality_score = self.calculate_data_quality(data)
                recommendations = self.generate_recommendations(data_quality_score)


        elif isinstance(data, str):
            logging.info("DataAnalyticsAgent processing unstructured text.")
            # For unstructured text, just get a response from Gemini
            response = self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)
            # No specific data quality or insights for generic text by default here.

        else:
            logging.warning(f"Unsupported data type for DataAnalyticsAgent: {type(data)}")
            response = "I cannot perform analysis on the provided data type."

        return {
            "response": response,
            "key_insights": key_insights,
            "data_quality_score": data_quality_score,
            "recommendations": recommendations
        }

    def _execute_dataframe_query(self, data: pd.DataFrame, query: str) -> Union[pd.DataFrame, str]:
        """
        Attempts to parse a natural language query and execute a basic
        Pandas DataFrame operation (filtering, column selection).

        This is a simplified NLP-to-code mapping and can be expanded.
        """
        query_lower = query.lower()
        
        # Example: "which strikers has overall rating more than 85. mention their name and overall rating."
        # This regex is specific for this type of query. More patterns can be added.
        match = re.search(r"(?:which|list|show me) (?:the )?(\w+)\s+(?:has|have|with|where) (.+?) (?:more than|greater than|over|above|less than|below|under)\s+(\d+)", query_lower)
        
        if match:
            target_group_phrase = match.group(1).strip() # e.g., "strikers"
            metric_phrase = match.group(2).strip()       # e.g., "overall rating"
            threshold_value = float(match.group(3))      # e.g., 85.0

            logging.info(f"Detected query: target_group='{target_group_phrase}', metric='{metric_phrase}', threshold='{threshold_value}'")

            # --- Column Mapping ---
            # Define mappings for common column names or phrases
            col_map = {
                'positions': ['positions', 'position', 'role'],
                'overall_rating': ['overall rating', 'rating', 'overall_rating', 'overall'],
                'full_name': ['name', 'full name', 'player name', 'player'],
                'club': ['club', 'team']
            }

            # Find actual column names in the DataFrame, prioritizing based on map
            actual_pos_col = next((col for col_options in col_map['positions'] for col in data.columns if col_options in col.lower()), None)
            actual_metric_col = next((col for col_options in col_map['overall_rating'] for col in data.columns if col_options in col.lower()), None)
            actual_name_col = next((col for col_options in col_map['full_name'] for col in data.columns if col_options in col.lower()), None)
            
            # Fallback for name column if not found
            if actual_name_col is None and 'name' in data.columns:
                actual_name_col = 'name'
            elif actual_name_col is None and 'full_name' in data.columns:
                actual_name_col = 'full_name'


            if not actual_metric_col:
                return f"I couldn't identify the rating/metric column '{metric_phrase}' in your data. Please ensure it's present and named clearly."
            
            filtered_df = data.copy()

            # --- Apply Filtering based on detected phrases ---
            # Filter by 'striker' position if applicable
            if "striker" in target_group_phrase or "strikers" in target_group_phrase:
                if actual_pos_col and actual_pos_col in filtered_df.columns:
                    # Case-insensitive check for 'ST' within the positions string
                    filtered_df = filtered_df[filtered_df[actual_pos_col].astype(str).str.contains('ST', case=False, na=False)]
                    logging.info(f"Filtered by position: {actual_pos_col} contains 'ST'.")
                else:
                    logging.warning(f"Position column '{actual_pos_col}' not found or cannot be filtered for 'striker'.")
                    # Continue without this specific filter, but inform the user later

            # Apply numerical filter based on metric and threshold
            if actual_metric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[actual_metric_col]):
                if "more than" in query_lower or "greater than" in query_lower or "over" in query_lower or "above" in query_lower:
                    filtered_df = filtered_df[filtered_df[actual_metric_col] > threshold_value]
                    logging.info(f"Filtered by {actual_metric_col} > {threshold_value}.")
                elif "less than" in query_lower or "below" in query_lower or "under" in query_lower:
                    filtered_df = filtered_df[filtered_df[actual_metric_col] < threshold_value]
                    logging.info(f"Filtered by {actual_metric_col} < {threshold_value}.")
            else:
                return f"The column '{metric_phrase}' (identified as '{actual_metric_col}') is not numeric, so I cannot filter it by value."

            # --- Select and Order Output Columns ---
            output_cols = []
            if actual_name_col and actual_name_col in filtered_df.columns: 
                output_cols.append(actual_name_col)
            if actual_metric_col and actual_metric_col in filtered_df.columns: 
                output_cols.append(actual_metric_col)
            
            # If the user asks to "mention their name and overall rating"
            if "mention their name and overall rating" in query_lower:
                if actual_name_col not in output_cols and actual_name_col and actual_name_col in filtered_df.columns:
                    output_cols.insert(0, actual_name_col) # Add name as first column
                if actual_metric_col not in output_cols and actual_metric_col and actual_metric_col in filtered_df.columns:
                    output_cols.append(actual_metric_col) # Add metric as second column

            if not output_cols:
                # If no specific output columns are requested, default to all columns of the filtered data
                logging.warning("No specific output columns requested. Returning all columns of filtered data.")
                output_cols = filtered_df.columns.tolist()
                if not output_cols: # If filtered_df is also empty
                    return "No matching records found based on your query or relevant columns could not be determined for display."
                
            # Ensure only columns present in filtered_df are in output_cols
            output_cols = [col for col in output_cols if col in filtered_df.columns]

            if not filtered_df.empty and output_cols:
                # Sort by the metric column in descending order
                if actual_metric_col and actual_metric_col in output_cols:
                    filtered_df = filtered_df.sort_values(by=actual_metric_col, ascending=False)
                
                # Return the head (e.g., top 5) if the query implies a limited list
                if "top 5" in query_lower or "list 5" in query_lower:
                    return filtered_df[output_cols].head(5)
                else:
                    # Limit general results to avoid excessively large outputs in chat
                    return filtered_df[output_cols].head(10) # Default to top 10 if no specific limit
            else:
                return "No matching records found based on your criteria. Please refine your query."
        
        logging.info("No specific DataFrame query pattern matched. Proceeding with general analysis.")
        return "I can analyze the data generally, but your specific data query could not be executed directly. Perhaps you could rephrase or ask for a summary of relevant columns?"


    def extract_key_insights(self, data: pd.DataFrame) -> List[str]:
        """
        Extracts key insights from a Pandas DataFrame.
        """
        insights = []
        
        if not isinstance(data, pd.DataFrame) or data.empty:
            return ["No data to extract insights from."]

        insights.append(f"Dataset contains {data.shape[0]} records and {data.shape[1]} columns.")

        missing_data = data.isnull().sum()
        high_missing = missing_data[missing_data > 0.1 * data.shape[0]]
        if not high_missing.empty:
            insights.append(f"High missing data in columns: {', '.join(high_missing.index.tolist())}")

        # Correlation insights for numeric data
        numeric_cols = data.select_dtypes(include=[np.number])
        if not numeric_cols.empty and len(numeric_cols.columns) > 1:
            try:
                corr_matrix = numeric_cols.corr()
                # Find high correlations (e.g., > 0.7 or < -0.7, excluding self-correlation)
                high_corr = corr_matrix[((corr_matrix > 0.7) | (corr_matrix < -0.7)) & (corr_matrix != 1.0)].stack()
                if not high_corr.empty:
                    corr_insights = []
                    for (col1, col2), value in high_corr.items():
                        corr_insights.append(f"Strong {'positive' if value > 0 else 'negative'} correlation ({value:.2f}) between {col1} and {col2}.")
                    insights.append("Identified strong correlations:\n" + "\n".join(corr_insights[:3])) # Limit to top 3
            except Exception as e:
                logging.warning(f"Could not calculate correlations: {e}")

        # Outlier detection (simple IQR method for numeric columns)
        for col in numeric_cols.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if not outliers.empty:
                insights.append(f"Potential outliers detected in '{col}' (e.g., {outliers[col].min()} and {outliers[col].max()}).")

        # High cardinality in categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist() # Define categorical_cols here
        for col in categorical_cols:
            unique_count = data[col].nunique()
            if unique_count > 0.5 * data.shape[0] and unique_count > 10: # More than 50% unique or high count
                insights.append(f"High cardinality in categorical column '{col}' with {unique_count} unique values.")

        return insights

    def calculate_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates data quality metrics for a Pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            return {"completeness": 0.0, "consistency": 0.0, "validity": 0.0}

        completeness = 1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        consistency = 1 - data.duplicated().sum() / data.shape[0]
        
        # Basic validity check (can be expanded with specific rules)
        validity_scores = []
        for col in data.columns:
            validity_scores.append(self.check_validity(data[col]))
        validity = np.mean(validity_scores)

        return {
            "completeness": round(completeness * 100, 2),
            "consistency": round(consistency * 100, 2),
            "validity": round(validity * 100, 2)
        }

    def check_validity(self, column: pd.Series) -> float:
        """
        Performs a basic validity check for a single column.
        """
        if column.empty:
            return 1.0 # Or 0.0 depending on how empty columns should impact validity

        if column.dtype == 'object':
            # Check for non-empty strings and basic alphanumeric patterns
            valid_strings = column.astype(str).str.strip() != ''
            # Adding a basic regex check for alphanumeric, can be more complex
            # valid_strings = valid_strings & column.astype(str).str.match(r'^[a-zA-Z0-9\s\-\_.,]+$').fillna(False)
            return valid_strings.mean() # Proportion of valid strings
        elif column.dtype in ['int64', 'float64']:
            # For numeric types, primarily check for non-null values
            return column.notnull().mean()
        else:
            # For other types (e.g., datetime, boolean), assume 100% validity if not null
            return column.notnull().mean()

    def generate_recommendations(self, score: Dict[str, float]) -> List[str]:
        """
        Generates data quality recommendations based on the calculated scores.
        """
        recommendations = []
        if score.get('completeness', 100) < 90:
            recommendations.append("Fill missing values to improve completeness. Consider imputation strategies like mean, median, mode, or more advanced methods.")
        if score.get('consistency', 100) < 90:
            recommendations.append("Remove duplicate rows or standardize inconsistent entries to improve consistency.")
        if score.get('validity', 100) < 90:
            recommendations.append("Review data formats and enforce data validation rules to improve validity. This may involve cleaning messy text or correcting numerical types.")
        
        if not recommendations:
            recommendations.append("Data quality appears to be high. Keep up the good work!")
            
        return recommendations
