# A RAG based pdf chatbot
- HuggingFaceEmbeddings for vector embeddings
- FAISS for vecor database
- Llama through Groq for Llm

# Clone the repo
```
git clone https://github.com/ambareeshav/rag-pdf-chatbot

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
