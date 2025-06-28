import streamlit as st
import psycopg2
import pandas as pd

class DBConnector:
    """
    Handles connections to the PostgreSQL database using Streamlit secrets.
    """
    def __init__(self):
        try:
            self.conn = self._get_connection()
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            self.conn = None

    @st.cache_resource
    def _get_connection(_self):
        """Caches the database connection."""
        # This line reads the [postgres] section from your secrets.toml file
        # and passes all the key-value pairs (host, port, dbname, etc.)
        # directly to the connection function.
        return psycopg2.connect(**st.secrets["postgres"])

    def fetch_data(self, query: str) -> pd.DataFrame:
        """Fetches data from the database and returns a pandas DataFrame."""
        if not self.conn:
            raise ConnectionError("No active database connection.")
        
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            # Attempt to reconnect on failure
            self.conn = self._get_connection()
            df = pd.read_sql_query(query, self.conn)
            return df
