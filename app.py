import streamlit as st
import pandas as pd
from dual_rag_agent_system import AgentCoordinator
from utils.file_processor import FileProcessor
from utils.data_handler import DataHandler
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gemini API Key Authentication Flow ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if st.session_state.api_key is None:
    st.title("🚀 Welcome to the Multi-Agent Analytics Platform")
    st.header("Gemini API Key Required")
    st.markdown(
        "To use the Gemini-powered agents, please enter your Google Gemini API key. "
        "This is a one-time setup per session."
    )
    with st.expander("How to get your Gemini API Key"):
        st.markdown("""
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
            2. Create a new API key or use an existing one.
            3. Copy the generated API key.
            4. Ensure your project has access to the Gemini API.
        """)
    api_input = st.text_input(
        "Paste your Gemini API Key here:",
        type="password",
        help="Your API key is stored securely for this session only and is not saved anywhere."
    )
    if st.button("Authenticate and Start Application"):
        if api_input:
            st.session_state.api_key = api_input
            st.rerun()
        else:
            st.error("Please enter a valid Gemini API key.")
    st.stop()

# --- Main Application Logic (runs only after token is provided) ---
if 'coordinator' not in st.session_state:
    with st.spinner("🚀 Initializing AI Agents... This may take a moment."):
        try:
            # Pass the Gemini API key to the AgentCoordinator
            st.session_state.coordinator = AgentCoordinator(api_key=st.session_state.api_key)
        except Exception as e:
            st.error(f"Failed to initialize agents. Error: {e}")
            st.session_state.api_key = None # Clear API key to force re-authentication
            if st.button("Try Again"):
                st.rerun()
            st.stop()

if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

st.title("🤖 Multi-Agent Data Analytics & Visualization")
st.markdown("Upload your data and let AI agents provide intelligent insights and visualizations.")

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'txt', 'pdf', 'docx'], # Added xlsx and txt based on file_processor
        help="Upload CSV, XLSX, TXT, DOCX, or PDF files for analysis"
    )

    if uploaded_file is not None:
        if st.session_state.get('coordinator'):
            # Clear context only if a new file is uploaded
            # Note: The AgentCoordinator itself doesn't have a clear_context method directly.
            # This would imply resetting internal states if necessary.
            # For this setup, simply re-processing the file and resetting session_state.data is sufficient.
            pass 

        # Correctly initialize FileProcessor with the uploaded_file
        file_processor = FileProcessor(uploaded_file)
        with st.spinner("Processing file..."):
            try:
                processed_data = file_processor.process_file() # Corrected method name
                if processed_data is not None:
                    st.session_state.chat_history = [] # Reset chat history on new file upload
                    st.session_state.analysis_results = None # Reset analysis results
                    st.session_state.data = processed_data
                    st.success("✅ File processed successfully!")

                    # Display data summary in sidebar using DataHandler
                    data_handler = DataHandler()
                    summary = data_handler.summarize(processed_data)
                    st.sidebar.subheader("📊 Data Summary")
                    st.sidebar.json(summary) # Display summary as JSON for detailed view

                else:
                    st.error("❌ Failed to process file. The format may be unsupported or the file corrupted.")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
            
    # Display file information if data is loaded
    if st.session_state.data is not None:
        st.subheader("📋 File Information")
        if isinstance(st.session_state.data, pd.DataFrame):
            st.write(f"**Shape:** {st.session_state.data.shape}")
            st.write(f"**Columns:** {len(st.session_state.data.columns)}")
            st.write(f"**Memory Usage (MB):** {st.session_state.data.memory_usage(deep=True).sum() / (1024*1024):.2f}")
        else:
            st.write(f"**Type:** Text document")
            st.write(f"**Length:** {len(str(st.session_state.data))} characters")

    st.divider()
    st.header("🔧 Agent Status")
    # Assuming AgentCoordinator has a method to get agent status (mocked or real)
    try:
        # If AgentCoordinator needs specific logic to get status, implement it there.
        # For now, we can show a placeholder or basic status.
        st.markdown(f"**Analytics Agent:** `Ready`")
        st.markdown(f"**Visualization Agent:** `Ready`")
        st.markdown(f"**Gemini Agent:** `Ready`")
    except Exception as e:
        st.error(f"Could not retrieve agent status: {e}")


# --- Main Content Area ---
if st.session_state.data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "🔍 Analytics", "📈 Visualizations", "💬 Chat Interface"])

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
        st.header("🔍 Data Analytics Agent")
        if st.button("🚀 Generate Analytics Report", type="primary"):
            with st.spinner("Analytics agent is processing your data..."):
                try:
                    # Directly call the analytics agent's analyze_data method
                    # This returns the structured analysis results directly
                    analysis_results = st.session_state.coordinator.analytics_agent.analyze_data(
                        st.session_state.data, 
                        "Generate a detailed analytics report including key insights, data quality, and recommendations.",
                        st.session_state.chat_history # Pass chat history for context
                    )
                    st.session_state.analysis_results = analysis_results
                except Exception as e:
                    st.error(f"❌ Error generating analytics: {str(e)}")
        if st.session_state.analysis_results:
            st.subheader("📋 Analysis Results")
            results = st.session_state.analysis_results
            
            # Access the structured results from the analytics_agent
            if 'response' in results: # This is the Gemini agent's text response summarizing insights
                st.subheader("AI-Generated Summary")
                st.markdown(results['response'])

            if 'key_insights' in results and results['key_insights']:
                st.subheader("💡 Key Insights")
                for insight in results['key_insights']: st.markdown(f"• {insight}")
            if 'data_quality_score' in results:
                st.subheader("🔍 Data Quality Assessment")
                quality = results['data_quality_score']
                if isinstance(quality, dict):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Completeness", f"{quality.get('completeness', 0):.1f}%")
                    c2.metric("Consistency", f"{quality.get('consistency', 0):.1f}%")
                    c3.metric("Validity", f"{quality.get('validity', 0):.1f}%")
            if 'recommendations' in results and results['recommendations']:
                st.subheader("📝 Recommendations")
                for rec in results['recommendations']: st.markdown(f"• {rec}")
        else:
            st.info("Click 'Generate Analytics Report' to see insights here.")


    with tab3:
        st.header("📈 Data Visualization Agent")
        viz_query = st.text_input("🗣️ Describe the visualization you want:", placeholder="e.g., 'Create a bar chart showing sales by region'")
        if st.button("🎨 Generate Visualization", type="primary") and viz_query:
            with st.spinner("Visualization agent is creating your chart..."):
                try:
                    # Call handle_chat_query which returns both response and chart
                    response_text, chart_fig = st.session_state.coordinator.handle_chat_query(
                        st.session_state.data, 
                        viz_query, 
                        st.session_state.chat_history
                    )
                    
                    if chart_fig: # Check if a chart was actually generated
                        st.subheader("📊 Generated Visualization")
                        st.plotly_chart(chart_fig, use_container_width=True)
                        st.info(f"**Chart Explanation:** {response_text}") # Use response_text as explanation
                    else:
                        st.warning("Could not generate a visualization based on your query or data type.")
                        st.info(f"Agent's response: {response_text}") # Show text response even if no chart
                except Exception as e:
                    st.error(f"❌ Error generating visualization: {str(e)}")

    with tab4:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input at the bottom
        if user_query := st.chat_input("Ask questions about your data..."):
            # Add user message to history immediately for display
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Agents are analyzing your data..."):
                    try:
                        # Call handle_chat_query for general chat, it will return response and potentially a chart (which is not displayed here)
                        response, _ = st.session_state.coordinator.handle_chat_query(
                            st.session_state.data,
                            user_query,
                            st.session_state.chat_history
                        )
                        st.markdown(response)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    except Exception as e:
                        error_message = f"Sorry, an error occurred: {e}"
                        st.error(error_message)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': error_message})
else:
    st.header("Please Upload a File")
    st.markdown("Use the sidebar on the left to upload a CSV, XLSX, TXT, DOCX, or PDF file to begin the analysis.")
