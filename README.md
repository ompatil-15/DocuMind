# DocuMind ⚡📄🤖

DocuMind is a document-interaction assistant powered by Google Generative AI. It allows users to upload PDF files, extract the text, and ask questions based on the document content. It provides various AI-powered modes, such as legal, financial, and general document assistance, all accessible via a simple Streamlit interface.

[Live Website](https://ompatil-genai-rag.streamlit.app/)

## Features

- **PDF Uploading & Text Extraction**: Upload PDF files, and the app extracts and processes the text for quick question-answering.
- **Vector-based Retrieval**: Uses FAISS for fast and efficient similarity search over embedded document chunks.
- **Chat History**: Keeps track of chat history to provide context-aware question-answering.
- **Pre-trained Modes**: Includes pre-trained AI modes:
  - Regular Assistant
  - Plant Expert
  - Legal Assistant
  - Finance Master
- **Easy Reset**: Clear uploaded documents and reset stored embeddings.
- **Google Generative AI Integration**: Uses Google Generative AI for both embedding and answering questions.

## Tech Stack

- **Streamlit**: For building the web interface.
- **Google Generative AI**: For embeddings and question-answering models.
- **FAISS**: For vector-based document retrieval.
- **LangChain**: For chaining document retrieval and question-answering functionalities.
- **PyPDF2**: For PDF text extraction.
- **Docker**: Containerized for easy deployment and reproducibility.


