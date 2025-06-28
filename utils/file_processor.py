# utils/file_processor.py

import pandas as pd
import PyPDF2
import docx
import io
from typing import Union, Optional

class FileProcessor:
    """
    Utility class for processing different file types (CSV, DOC, DOCX, PDF).
    """

    def __init__(self):
        self.supported_formats = ['csv', 'doc', 'docx', 'pdf']

    # --- THIS IS THE FIX ---
    # Rename this method from 'process_file' to 'process'
    def process(self, uploaded_file) -> Optional[Union[pd.DataFrame, str]]:
    # --- END OF FIX ---
        """
        Process uploaded file based on its extension.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame for CSV files, str for document files, None on error
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            if file_extension == 'csv':
                return self._process_csv(uploaded_file)
            elif file_extension in ['doc', 'docx']:
                return self._process_word_document(uploaded_file)
            elif file_extension == 'pdf':
                return self._process_pdf(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    def _process_csv(self, uploaded_file) -> pd.DataFrame:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            if df.empty:
                raise ValueError("CSV file is empty")
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")

    def _process_word_document(self, uploaded_file) -> str:
        try:
            doc = docx.Document(uploaded_file)
            text_content = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(text_content)
            if not full_text.strip():
                raise ValueError("No text content found in DOCX file")
            return full_text
        except Exception as e:
            raise Exception(f"Error processing Word document: {str(e)}")

    def _process_pdf(self, uploaded_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            full_text = "\n\n".join(text_content)
            if not full_text.strip():
                raise ValueError("No text content found in PDF file")
            return full_text
        except Exception as e:
            raise Exception(f"Error processing PDF file: {str(e)}")
