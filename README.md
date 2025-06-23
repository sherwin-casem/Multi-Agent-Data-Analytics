📊 Multi-Agent Data Analytics & RAG System 🤖
Welcome to the Multi-Agent Data Analytics & RAG System! This project provides an intelligent platform for analyzing both structured and unstructured data using a powerful multi-agent architecture, Retrieval-Augmented Generation (RAG), and Google's Gemini models.

✨ Features
📈 Data Analysis: Summarize datasets, extract key insights, and assess data quality.

📊 Dynamic Visualizations: Generate interactive charts (bar, line, scatter, pie, histogram, heatmap, etc.) based on natural language queries.

🧠 Intelligent Q&A: Leverage Google's Gemini 1.5 Flash LLM for comprehensive answers grounded in your data.

📚 Retrieval-Augmented Generation (RAG): Enhance LLM responses by retrieving relevant information from your uploaded documents using LangChain and ChromaDB.

📄 Multi-format File Support: Process CSV, XLSX, PDF, DOCX, and TXT files.

🗣️ Conversational Interface: Interact with the system using natural language via a Streamlit chat interface.

🚀 Technology Stack
This project is built primarily with Python and leverages the following key technologies:

Framework:

Streamlit: For building the interactive web application interface (app.py).

Large Language Models (LLM) & Embeddings:

Google Gemini 1.5 Flash: The core LLM for generating responses, insights, and explanations (agents/gemini_agent.py).

GoogleGenerativeAIEmbeddings: For creating vector embeddings of text.

RAG & Orchestration:

LangChain: Framework for building LLM applications, used for RAG pipeline orchestration, text splitting, and managing LLM/embedding integrations (dual_rag_agent_system.py).

ChromaDB: A lightweight vector database for storing and querying text embeddings for efficient information retrieval (dual_rag_agent_system.py).

Data Handling & Analysis:

Pandas: For structured data manipulation and analysis.

NumPy: For numerical operations.

Plotly: For creating interactive data visualizations (utils/chart_generator.py).

File Processing:

PyMuPDF (fitz): For PDF file processing.

python-docx: For DOCX file processing.

Custom Agents:

AgentCoordinator: Orchestrates the flow between different agents and RAG components (dual_rag_agent_system.py).

GeminiAgent: Handles direct interactions with the Gemini LLM (agents/gemini_agent.py).

DataAnalyticsAgent: Focuses on data summarization, insights, and quality assessment (agents/data_analytics_agent.py).

VisualizationAgent: Manages chart generation and explanation (agents/visualization_agent.py).

FileProcessor: Utility for handling various file uploads (utils/file_processor.py).

DataHandler: Utility for data preprocessing and summarization (utils/data_handler.py).

ChartGenerator: Utility for creating Plotly charts (utils/chart_generator.py).

🛠️ Setup & Installation
To get this project up and running locally, follow these steps:

Clone the Repository:

git clone <your-repository-url>
cd <your-repository-name>

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

Install Dependencies:
Create a pyproject.toml or requirements.txt file (as discussed in previous steps). If you have pyproject.toml, you can use uv or pip with pip install -e . or pip install -r requirements.txt.

Using pyproject.toml (recommended with uv):
Ensure uv is installed (pip install uv).

uv pip install -e .

Or, if using requirements.txt:

pip install -r requirements.txt

Make sure all required libraries including google-generativeai, langchain-google-genai, langchain-community, chromadb, PyMuPDF, python-docx are listed.

Set up your Google Gemini API Key:

Obtain your API key from Google AI Studio.

Do NOT hardcode your API key in the code.

Create a .env file in the root directory of your project (same level as app.py) and add your API key:

GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Alternatively, when deploying to Streamlit Cloud, you'll use Streamlit's built-in secrets management (as explained in previous steps).

In your app.py ensure you are loading the API key securely, e.g., using os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY").

▶️ How to Run the Application
Once setup is complete, you can run the Streamlit application:

streamlit run app.py

This will open the application in your default web browser (usually at http://localhost:8501).

🚀 How to Use
Upload Your File: On the sidebar, use the file uploader to select your CSV, XLSX, PDF, DOCX, or TXT file.

Authenticate (if prompted): Enter your Google Gemini API key if you haven't set it up as an environment variable or secret.

Chat with the Agents: In the main chat interface, ask questions about your data.

For structured data (CSV/XLSX): You can ask for summaries, insights, specific data points, or request visualizations.

Example: "Summarize the dataset."

Example: "Show me the distribution of age."

Example: "Which strikers has overall rating more than 85? mention their name and overall rating."

Example: "Plot a bar chart of sales by region."

For unstructured data (PDF/DOCX/TXT): You can ask questions about the document's content. The RAG system will retrieve relevant information.

Example: "What is the main topic of this document?"

Example: "Summarize the key findings."

View Results: The system will provide textual responses and, for structured data, interactive charts.

🤝 Contributions
Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request on the GitHub repository.

Enjoy exploring your data with the power of AI! ✨
