# RAG-PDF-Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that enables users to interact with PDF documents by asking questions and receiving contextually relevant answers. This project utilizes HuggingFace embeddings, FAISS for vector storage, and the Llama language model through Groq.

## Features

- **PDF Processing**: Extracts text content from PDF files.
- **Vector Embeddings**: Generates embeddings using HuggingFace's `all-MiniLM-L6-v2` model.
- **Vector Storage**: Stores embeddings in a FAISS vector database for efficient retrieval.
- **Conversational Interface**: Employs the Llama language model via Groq to generate responses based on user queries and document context.

## Prerequisites

- Python 3.8 or higher
- Groq API Key: Sign up at [Groq](https://groq.com/) to obtain your API key.

## Installation

# Clone the Repository
```
   git clone https://github.com/ambareeshav/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
```

# Install required libraries
```
pip install -r requirements.txt
```

# Set API key for Groq
```
setx GROQ_API_KEY "gsk_..."
```

Run **data.py**

- Process the pdf
- Generate vector embeddings using HuggingFaceEmbedding
- Store embedded data in a FAISS vector db
- Save the db locally


Run **app.py**

- Load the db
- create template, prompt and chain using llama via Groq.
- Invoke the chain by passing context, question and chat history.
