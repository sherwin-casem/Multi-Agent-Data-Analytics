import pandas as pd
import fitz  # PyMuPDF
import docx
import streamlit as st # Only for type hinting UploadedFile

class FileProcessor:
    """
    Handles file uploads and processing for CSV, XLSX, PDF, DOCX, and TXT formats.
    """

    def __init__(self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile):
        if uploaded_file is None:
            raise ValueError("Uploaded file cannot be None.")
        self.uploaded_file = uploaded_file
        self.file_type = uploaded_file.name.split('.')[-1].lower()

    def process_file(self):
        """
        Processes the uploaded file based on its type.
        """
        if self.file_type == 'csv':
            return pd.read_csv(self.uploaded_file)
        elif self.file_type == 'xlsx':
            return pd.read_excel(self.uploaded_file, engine='openpyxl')
        elif self.file_type == 'txt':
            return self.uploaded_file.read().decode('utf-8')
        elif self.file_type == 'pdf':
            doc = fitz.open(stream=self.uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif self.file_type == 'docx':
            doc = docx.Document(self.uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error(f"Unsupported file type: {self.file_type}")
            return None