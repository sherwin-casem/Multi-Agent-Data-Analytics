import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataHandler:
    """
    Handles preprocessing and summarization for structured and unstructured data.
    """

    def preprocess(self, data: Union[pd.DataFrame, str], options: Dict[str, Any] = None) -> Union[pd.DataFrame, str]:
        """
        Preprocesses the input data based on the provided options.

        Args:
            data (Union[pd.DataFrame, str]): The input data (DataFrame or string).
            options (Dict[str, Any], optional): Dictionary of preprocessing options. Defaults to None.

        Returns:
            Union[pd.DataFrame, str]: The preprocessed data.
        """
        if options is None:
            options = {}

        if isinstance(data, pd.DataFrame):
            logging.info("Preprocessing DataFrame data.")
            df = data.copy()
            try:
                if options.get('dropna'):
                    initial_rows = df.shape[0]
                    df = df.dropna()
                    logging.info(f"Dropped {initial_rows - df.shape[0]} rows with missing values.")
                if options.get('drop_duplicates'):
                    initial_rows = df.shape[0]
                    df = df.drop_duplicates()
                    logging.info(f"Dropped {initial_rows - df.shape[0]} duplicate rows.")
                if options.get('clean_columns'):
                    original_columns = list(df.columns)
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                    logging.info(f"Cleaned column names. Example: {original_columns[0]} -> {df.columns[0]}")
                return df
            except Exception as e:
                logging.error(f"Error during DataFrame preprocessing: {e}", exc_info=True)
                # Return original data if preprocessing fails to prevent data loss
                return data 

        elif isinstance(data, str):
            logging.info("Preprocessing string data.")
            text = data
            try:
                if options.get('lowercase'):
                    text = text.lower()
                    logging.info("Converted text to lowercase.")
                if options.get('clean_whitespace'):
                    text = ' '.join(text.split())
                    logging.info("Cleaned extra whitespace from text.")
                return text
            except Exception as e:
                logging.error(f"Error during string preprocessing: {e}", exc_info=True)
                return data # Return original text if preprocessing fails

        else:
            logging.warning(f"Unsupported data type for preprocessing: {type(data)}. Returning original data.")
            return data

    def summarize(self, data: Union[pd.DataFrame, str]) -> Dict[str, Any]:
        """
        Generates a summary of the input data based on its type.

        Args:
            data (Union[pd.DataFrame, str]): The input data (DataFrame or string).

        Returns:
            Dict[str, Any]: A dictionary containing the data summary.
        """
        if isinstance(data, pd.DataFrame):
            logging.info("Summarizing DataFrame data.")
            if data.empty:
                logging.warning("Attempted to summarize an empty DataFrame.")
                return {
                    "shape": (0, 0),
                    "columns": [],
                    "missing_values": {},
                    "duplicates": 0,
                    "memory_usage_mb": 0.0,
                    "summary_warning": "DataFrame is empty, summary is limited."
                }
            try:
                return {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "missing_values": data.isnull().sum().to_dict(),
                    "duplicates": int(data.duplicated().sum()),
                    "memory_usage_mb": round(data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            except Exception as e:
                logging.error(f"Error summarizing DataFrame: {e}", exc_info=True)
                return {"error": f"Failed to summarize DataFrame: {e}"}

        elif isinstance(data, str):
            logging.info("Summarizing string data.")
            if not data.strip(): # Check for effectively empty string
                logging.warning("Attempted to summarize an empty string.")
                return {
                    "total_characters": 0,
                    "total_words": 0,
                    "top_words": {},
                    "summary_warning": "String is empty, summary is limited."
                }
            try:
                words = data.split()
                return {
                    "total_characters": len(data),
                    "total_words": len(words),
                    "top_words": dict(Counter(words).most_common(10)) if words else {} # Handle empty word list
                }
            except Exception as e:
                logging.error(f"Error summarizing string data: {e}", exc_info=True)
                return {"error": f"Failed to summarize string: {e}"}

        else:
            logging.error(f"Unsupported data type for summarization: {type(data)}. Cannot provide a summary.")
            return {"error": "Unsupported data type for summarization."}
