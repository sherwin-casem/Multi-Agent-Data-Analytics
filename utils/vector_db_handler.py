# utils/vector_db_handler.py (Definitive Fix)

import os
import uuid
import logging
from typing import List, Optional
from pydantic import SecretStr

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDBHandler:
    """
    Handles all interactions with the Chroma vector database.
    """
    def __init__(self, api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(api_key))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # --- THIS IS THE DEFINITIVE FIX ---
        # Create the database directory in /tmp, which is always writable.
        self.db_dir = "/tmp/chroma_db_data"
        os.makedirs(self.db_dir, exist_ok=True)
        # --- END OF FIX ---
        
        self.vectorstore: Optional[Chroma] = None

    def process_text(self, text_content: str, collection_prefix: str) -> None:
        """Processes text and stores it in a new Chroma collection."""
        collection_name = f"{collection_prefix}_{uuid.uuid5(uuid.NAMESPACE_DNS, text_content[:100]).hex}"
        documents = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text_content)]
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_dir, # Use the correct /tmp path here
                collection_name=collection_name
            )
            self.vectorstore.persist()
            logging.info(f"Text processed into Chroma collection: {collection_name} at {self.db_dir}")
        except Exception as e:
            logging.error(f"Error processing text for RAG: {e}", exc_info=True)
            self.vectorstore = None

    def get_context(self, query: str) -> List[str]:
        """Retrieves context chunks from the active vector store."""
        if not self.vectorstore:
            return []
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(query)
            return [doc.page_content for doc in retrieved_docs]
        except Exception as e:
            logging.error(f"Error retrieving RAG context: {e}", exc_info=True)
            return []
