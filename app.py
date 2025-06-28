# --- REPLACE THE ENTIRE CONTENT OF app.py WITH THIS ---

import streamlit as st
import pandas as pd
from agents.coordinator import AgentCoordinator
from utils.file_processor import FileProcessor
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Football Analytics Agent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Football Theme & Dark Mode ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode

dark_theme = """
<style>
    .stApp {
        background-image: linear-gradient(to right top, #1a1a1a, #2c2c2c, #3e3e3e, #525252, #666666);
    }
    .st-emotion-cache-16txtl3 { /* Adjust color for text if needed */
        color: #e0e0e0;
    }
    div[data-testid="stToolbar"] {
        display: none; /* Hides the Streamlit toolbar */
    }
</style>
"""
light_theme = "<style>.stApp { background-color: #FAFAFA; }</style>"
st.markdown(dark_theme if st.session_state.dark_mode else light_theme, unsafe_allow_html=True)


# --- Login Flow ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("⚽ Football Analytics Agent Login")
    with st.form("login_form"):
        # Dummy values can be pre-filled for ease of use in development
        st.text_input("Username", key="username", value="admin")
        st.text_input("Password", type="password", key="password", value="a_strong_password")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Required for the LLM agent.")
        submitted = st.form_submit_button("Login")

        if submitted:
            if st.session_state.username and st.session_state.password and gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Please fill in all fields.")

if not st.session_state.logged_in:
    login_page()
    st.stop()


# --- Agent and Data Initialization (Cached) ---
@st.cache_resource
def initialize_coordinator(api_key):
    """Initializes the AgentCoordinator and caches it for the session."""
    try:
        return AgentCoordinator(gemini_api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize agents. Please check your API key. Error: {e}")
        return None

coordinator = initialize_coordinator(st.session_state.gemini_api_key)
if coordinator is None:
    st.stop()

@st.cache_data
def process_uploaded_file(uploaded_file):
    """Processes an uploaded file and caches the result."""
    file_processor = FileProcessor()
    try:
        return file_processor.process(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None

# --- Session State Management ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://as2.ftcdn.net/v2/jpg/04/17/36/11/1000_F_417361125_RnrhT3Np0zB0UpeD7QlwuOoyghEGGjBX.jpg", use_container_width=True)
    st.header("📁 Data Source")

    data_source = st.radio(
        "Choose a dataset to analyze:",
        ("EPL Match Data", "UCL Match Data", "Upload a Custom File")
    )

    data_loaded = False
    if data_source == "EPL Match Data":
        if st.button("Load EPL Data", type="primary"):
            with st.spinner("Loading EPL match data from database..."):
                st.session_state.data = coordinator.get_data_from_db("SELECT * FROM epl_match")
                st.session_state.chat_history = []
                data_loaded = True

    elif data_source == "UCL Match Data":
        if st.button("Load UCL Data", type="primary"):
            with st.spinner("Loading UCL match data from database..."):
                st.session_state.data = coordinator.get_data_from_db("SELECT * FROM ucl_matches")
                st.session_state.chat_history = []
                data_loaded = True

    elif data_source == "Upload a Custom File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'doc', 'docx', 'pdf'],
            help="Upload match data, player stats, or scouting reports."
        )
        if uploaded_file:
            with st.spinner("Processing file..."):
                st.session_state.data = process_uploaded_file(uploaded_file)
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []
                if st.session_state.data is not None:
                    data_loaded = True

    if data_loaded:
        st.success("✅ Data loaded successfully!")

    if st.session_state.data is not None:
        st.subheader("📋 Active Dataset Information")
        if isinstance(st.session_state.data, pd.DataFrame):
            st.write(f"**Shape:** {st.session_state.data.shape}")
            st.write(f"**Columns:** {len(st.session_state.data.columns)}")
        else:
            st.write(f"**Type:** Text document")
            st.write(f"**Length:** {len(str(st.session_state.data))} characters")

    st.divider()
    st.header("⚙️ Settings")
    if st.button("Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# --- Main Content Area ---
st.title("🤖 Football Analytics Agent")
st.markdown("Select a dataset from the sidebar to begin your analysis.")

if st.session_state.data is not None:
    tab_titles = ["📊 Data Preview", "🔬 Metadata Analysis", "🔍 Analytics", "📈 Visualizations", "💬 Chat"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        st.header("📊 Data Preview")
        if isinstance(st.session_state.data, pd.DataFrame):
            st.subheader("Raw Data")
            st.dataframe(st.session_state.data.head(100), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Basic Statistics")
                numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    st.dataframe(st.session_state.data[numeric_cols].describe())
                else:
                    st.info("No numeric columns found for statistics.")
            with col2:
                st.subheader("🔢 Data Types & Null Values")
                dtype_df = pd.DataFrame({
                    'Type': st.session_state.data.dtypes.astype(str),
                    'Non-Null Count': st.session_state.data.count(),
                    'Null Count': st.session_state.data.isnull().sum()
                }).reset_index()
                st.dataframe(dtype_df, use_container_width=True)
        else:
            st.subheader("Document Content")
            st.text_area("Content Preview", str(st.session_state.data)[:5000], height=400)

    with tab2:
        st.header("🔬 Metadata Analysis")
        if isinstance(st.session_state.data, pd.DataFrame):
            if st.button("Generate Metadata Report", type="primary"):
                with st.spinner("Generating detailed report..."):
                    pr = ProfileReport(st.session_state.data, title="Metadata Report", minimal=True)
                    st_profile_report(pr)
            else:
                st.info("Click the button to generate a detailed metadata and statistical report.")
        else:
            st.info("Metadata analysis is only available for tabular data (e.g., CSVs).")

    with tab3:
        st.header("🔍 Data Analytics Agent")
        if st.button("🚀 Generate Analytics Report", type="primary"):
            with st.spinner("Analytics agent is processing your data..."):
                analysis_results = coordinator.get_analytics_insights(st.session_state.data)
                st.session_state.analysis_results = analysis_results
        if st.session_state.analysis_results:
            st.subheader("📋 Analysis Results")
            st.write(st.session_state.analysis_results)

    with tab4:
        st.header("📈 Data Visualization Agent")
        viz_query = st.text_input("🗣️ Describe the visualization you want:", placeholder="e.g., 'Bar chart of goals by player'")
        if st.button("🎨 Generate Visualization", type="primary") and viz_query:
            with st.spinner("Visualization agent is creating your chart..."):
                chart_result = coordinator.generate_visualization(st.session_state.data, viz_query)
                # If chart_result is a tuple (chart, explanation), access by index
                if isinstance(chart_result, tuple) and len(chart_result) >= 1:
                    st.plotly_chart(chart_result[0], use_container_width=True)
                    if len(chart_result) > 1 and chart_result[1]:
                        st.info(chart_result[1])
                # If chart_result is a dict (legacy), fallback to old logic
                elif isinstance(chart_result, dict) and 'chart' in chart_result:
                    st.plotly_chart(chart_result['chart'], use_container_width=True)
                    if 'explanation' in chart_result:
                        st.info(chart_result['explanation'])
                else:
                    st.warning("Could not generate visualization. Try a different query.")

    with tab5:
        st.header("💬 Chat with your Data")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask about player stats, match events, etc..."):
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    response = coordinator.handle_chat_query(
                        st.session_state.data, user_query, st.session_state.chat_history
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})

else:
    st.header("Welcome to the Football Analytics Agent")
    st.markdown("Please select a dataset from the sidebar on the left to begin.")
