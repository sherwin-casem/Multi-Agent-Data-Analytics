import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from agents.gemini_agent import GeminiAgent  # Ensure this file is in the same directory
from utils.chart_generator import ChartGenerator  # Reuse chart logic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualizationAgent:
    """
    Agent responsible for generating visualizations based on user queries and data.
    It uses a GeminiAgent for explaining the visualizations.
    """
    def __init__(self, api_key: str):
        """
        Initializes the VisualizationAgent with a GeminiAgent and a ChartGenerator.

        Args:
            api_key (str): The API key for the GeminiAgent.
        """
        self.gemini_agent = GeminiAgent(api_key=api_key)
        self.chart_generator = ChartGenerator()
        logging.info("VisualizationAgent initialized.")

    def generate_chart(self, data: Union[pd.DataFrame, str, None], query: str, context: Any = None, chat_history: List[Dict[str, str]] = None) -> Tuple[Any, str]:
        """
        Generates a chart based on the query and data. If data is a DataFrame,
        it attempts to create a Plotly figure. If data is a string, it returns
        a message indicating charts are not applicable.

        Args:
            data (Union[pd.DataFrame, str, None]): The input data (DataFrame for charts, string for text, or None).
            query (str): The user's query describing the desired chart.
            context (Any, optional): Additional context for the GeminiAgent. Defaults to None.
            chat_history (List[Dict[str, str]], optional): Previous chat history for context. Defaults to None.

        Returns:
            Tuple[Any, str]: A tuple containing the Plotly figure (or None) and a text explanation.
        """
        if data is None:
            logging.warning("No data provided to VisualizationAgent.generate_chart. Cannot generate chart.")
            # Do NOT call gemini_agent.generate_response with None data
            return None, "No data available to generate a chart. Please upload a file first."
        
        if isinstance(data, pd.DataFrame):
            logging.info("Attempting to generate chart for DataFrame data.")
            chart_config = self.analyze_query(query, data)
            
            if chart_config.get("type") == "none": # Check if analyze_query decided against a chart
                reason = chart_config.get("reason", "No suitable chart type or columns found.")
                logging.info(f"Chart generation skipped based on analyze_query: {reason}")
                # Still generate an explanation using GeminiAgent if data is valid for it
                explanation = self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)
                return None, f"Could not generate a specific chart for the given query: {reason}. {explanation}"

            try:
                fig = self.chart_generator.create_chart(data, chart_config)
                # Explanation should still come from Gemini Agent, providing context on the generated chart
                explanation = self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)
                logging.info(f"Chart generated successfully: {chart_config.get('type')}")
                return fig, explanation
            except Exception as e:
                logging.error(f"Error creating chart in VisualizationAgent: {e}", exc_info=True)
                error_explanation = f"An error occurred while generating the chart: {e}. "
                # Only call gemini_agent if 'data' is a valid type for it (DataFrame or str)
                if isinstance(data, (pd.DataFrame, str)):
                    error_explanation += self.gemini_agent.generate_response(data, f"Explain why a chart could not be generated for '{query}' given the data structure and this error: {e}", context=context, chat_history=chat_history)
                else: # Fallback if data somehow becomes an unsupported type here
                    error_explanation += "Please check your data and query."
                return None, error_explanation
        
        elif isinstance(data, str):
            logging.info("Chart generation not applicable for unstructured text data.")
            # For unstructured text, still try to get a general response from Gemini
            explanation = self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)
            return None, f"Charts are typically generated for structured (tabular) data. This document is unstructured text. {explanation}"
        
        else:
            logging.warning(f"Unsupported data type for chart generation: {type(data)}. Cannot generate chart.")
            # Do NOT call gemini_agent.generate_response with unsupported data type
            return None, f"Unsupported data type ({type(data)}) for visualization. Charts require structured data (DataFrame) or specific text content."

    def analyze_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the user's query to determine the best chart configuration for a DataFrame.

        Args:
            query (str): The user's query about the visualization.
            data (pd.DataFrame): The DataFrame to be visualized.

        Returns:
            Dict[str, Any]: A dictionary containing the suggested chart type and parameters.
                            Returns a 'none' type if no suitable chart can be determined.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            logging.warning("Cannot analyze query for chart generation: Data is not a DataFrame or is empty.")
            return {"type": "none", "title": "Invalid Data for Charting", "reason": "Data is not a DataFrame or is empty."}

        query_lower = query.lower()
        chart_type = None # Start with None, determine explicitly
        
        # Simple keyword-based chart type detection
        if "line" in query_lower:
            chart_type = "line"
        elif "scatter" in query_lower:
            chart_type = "scatter"
        elif "histogram" in query_lower or "distribution" in query_lower:
            chart_type = "histogram"
        elif "pie" in query_lower or "proportion" in query_lower:
            chart_type = "pie"
        elif "box" in query_lower:
            chart_type = "box"
        elif "heatmap" in query_lower or "correlation" in query_lower:
            chart_type = "heatmap"
        elif "treemap" in query_lower:
            chart_type = "treemap"
        elif "area" in query_lower:
            chart_type = "area"
        elif "violin" in query_lower:
            chart_type = "violin"
        elif "sunburst" in query_lower:
            chart_type = "sunburst"
        elif "bar" in query_lower or "compare" in query_lower or "count" in query_lower: # Add more general bar triggers
            chart_type = "bar"
        logging.debug(f"Initial determined chart type: {chart_type} from query: {query}")

        columns = data.columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        x_col, y_col, names_col, values_col = None, None, None, None # Initialize all
        chart_title = ""
        reason = None

        # Helper to find column by name, prioritizing exact then case-insensitive match from query
        def find_col_in_query(query_text_lower, col_list):
            for c in col_list:
                if c.lower() in query_text_lower: # Check if column name (or part of it) is in query
                    return c
            return None

        # --- Intelligent Column Selection based on Chart Type ---
        if chart_type in ["bar", "line", "scatter", "area", "box", "violin"]:
            # Attempt to find columns from query
            x_col = find_col_in_query(query_lower, columns)
            y_col = find_col_in_query(query_lower, numeric_cols) # Prefer numeric for Y

            # Fallback if not found in query
            if x_col is None and columns: x_col = columns[0]
            if y_col is None and numeric_cols: y_col = numeric_cols[0]
            elif y_col is None and columns and len(columns) > 1: y_col = columns[1] # fallback to second col if no numeric

            if not x_col or not y_col:
                reason = f"{chart_type} chart requires both X and Y columns. Could not identify suitable columns from data or query."
                return {"type": "none", "title": "Missing Columns for Charting", "reason": reason}
            if not pd.api.types.is_numeric_dtype(data[y_col]):
                reason = f"The selected Y-column '{y_col}' for {chart_type} chart is not numeric. Please specify a numeric column."
                return {"type": "none", "title": "Invalid Y-axis Data Type", "reason": reason}
            chart_title = f"{chart_type.capitalize()} of {y_col} by {x_col}"

        elif chart_type == "histogram":
            x_col = find_col_in_query(query_lower, numeric_cols)
            if x_col is None and numeric_cols: x_col = numeric_cols[0] # Fallback to first numeric column

            if not x_col:
                reason = "Histogram requires a numeric X-axis column. Could not identify one."
                return {"type": "none", "title": "Missing X-axis Column", "reason": reason}
            if not pd.api.types.is_numeric_dtype(data[x_col]):
                reason = f"The selected X-column '{x_col}' for histogram is not numeric. Please specify a numeric column."
                return {"type": "none", "title": "Invalid X-axis Data Type", "reason": reason}
            chart_title = f"Distribution of {x_col}"

        elif chart_type in ["pie", "sunburst"]:
            names_col = find_col_in_query(query_lower, categorical_cols)
            if names_col is None and categorical_cols: names_col = categorical_cols[0]
            elif names_col is None and columns: names_col = columns[0] # Fallback to first available if no categorical

            # Check for explicit values column in query
            values_col = find_col_in_query(query_lower, numeric_cols)

            if not names_col:
                reason = f"{chart_type} chart requires a column for categories/names. Could not identify one."
                return {"type": "none", "title": "Missing Names Column", "reason": reason}
            
            # If values_col is specified and not numeric, it's an issue
            if values_col and not pd.api.types.is_numeric_dtype(data[values_col]):
                reason = f"The specified values column '{values_col}' for {chart_type} chart is not numeric. Please provide a numeric column."
                return {"type": "none", "title": "Invalid Values Column Type", "reason": reason}

            chart_title = f"Proportion by {names_col}"

        elif chart_type == "heatmap":
            if len(numeric_cols) < 2:
                reason = "Heatmap requires at least two numeric columns to show correlations."
                return {"type": "none", "title": "Insufficient Numeric Columns", "reason": reason}
            chart_title = "Correlation Heatmap"
        
        # If no specific chart type was determined or columns couldn't be found for the chosen type,
        # fallback to a default or return "none"
        if chart_type is None:
            # If no chart type was explicitly requested, try to suggest a bar chart if columns are suitable
            if categorical_cols and numeric_cols:
                chart_type = "bar"
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                chart_title = f"Default Bar Chart: {y_col} by {x_col}"
            elif numeric_cols: # If only numeric, suggest histogram
                chart_type = "histogram"
                x_col = numeric_cols[0]
                chart_title = f"Default Histogram: Distribution of {x_col}"
            elif categorical_cols: # If only categorical, suggest pie chart
                chart_type = "pie"
                names_col = categorical_cols[0]
                chart_title = f"Default Pie Chart: Proportion by {names_col}"
            else:
                reason = "Could not infer a suitable chart type or columns from your query and data. Please be more specific."
                return {"type": "none", "title": "No Suitable Chart Found", "reason": reason}

        # Final construction of chart_config_output
        chart_config_output = {
            "type": chart_type,
            "title": chart_title,
            "reason": reason # Will be None unless a specific issue was encountered
        }

        # Add columns based on the final determined chart type
        if chart_type in ["bar", "line", "scatter", "area", "box", "violin"]:
            chart_config_output["x_column"] = x_col
            chart_config_output["y_column"] = y_col
        elif chart_type == "histogram":
            chart_config_output["x_column"] = x_col
        elif chart_type in ["pie", "sunburst"]:
            chart_config_output["names_column"] = names_col
            if values_col: # Only add if a specific values_col was identified
                chart_config_output["values_column"] = values_col
        # Heatmap handles its own column selection internally based on numeric_cols

        logging.debug(f"Final analyzed query, chart_config: {chart_config_output}")
        return chart_config_output

    def explain_visualization(self, data: Union[pd.DataFrame, str], query: str = "Explain the chart", context: Any = None, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Uses the GeminiAgent to explain a visualization.

        Args:
            data (Union[pd.DataFrame, str]): The data related to the visualization.
            query (str): A query requesting explanation of the chart.
            context (Any, optional): Additional context for the GeminiAgent. Defaults to None.
            chat_history (List[Dict[str, str]], optional): Previous chat history for context. Defaults to None.

        Returns:
            str: The explanation generated by the GeminiAgent.
        """
        return self.gemini_agent.generate_response(data, query, context=context, chat_history=chat_history)

