import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = "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"
# Load the vectorstore
@st.cache_resource
def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    db = FAISS.load_local(
        "faiss_index", 
        embeddings,  
        allow_dangerous_deserialization=True
    )
    return db

# Create and configure chain
def create_chain(llm, retriever):
    rag_template = """
    Answer the Question based only on the following context:
    {context}
    Question: {question}
    The user will also provide feedback that you will check before each response and tailor the response to the user:
    {feedback}
    you will also have access to chat history:
    {chat_history}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    chain = (
        {
            "chat_history": itemgetter("chat_history"),
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "feedback": itemgetter("feedback")
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Initialize LLM and chain
@st.cache_resource
def initialize_chatbot():
    db = load_index()
    retriever = db.as_retriever()
    llm = ChatGroq(model_name='llama3-70b-8192')
    chain = create_chain(llm, retriever)
    return chain

# Streamlit UI
def main():
    st.set_page_config(page_title="Chatbot UI", layout="wide")
    st.title("Chatbot Interface")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'feedback' not in st.session_state:
        st.session_state.feedback = []
    if 'chain_initialized' not in st.session_state:
        st.session_state.chain_initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for feedback
    st.sidebar.title("Feedback")
    for i, feedback in enumerate(st.session_state.feedback):
        st.sidebar.text(f"{i+1}. {feedback}")

    # Main chat interface
    chat_container = st.container()

    # Initialize LLM and chain with loading state
    if not st.session_state.chain_initialized:
        with st.spinner("Initializing chatbot... This may take a moment."):
            chain = initialize_chatbot()
            st.session_state.chain = chain
            st.session_state.chain_initialized = True
        st.success("Chatbot initialized successfully!")

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    with st.container():
        user_input = st.text_input("You:", key="user_input")
        send_button = st.button("Send")

        if send_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chain.invoke({
                            "chat_history": st.session_state.chat_history,
                            "question": user_input,
                            "feedback": st.session_state.feedback
                        })
                        st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append((user_input, response))
            
            # Clear the input box after sending
            st.rerun()

    # Feedback input
    feedback = st.text_input("Provide feedback (optional):")
    if st.button("Submit Feedback"):
        if feedback:
            st.session_state.feedback.append(feedback)
            st.sidebar.success("Feedback submitted successfully!")

if __name__ == "__main__":
    main()