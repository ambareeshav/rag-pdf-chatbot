from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["GROQ_API_KEY"] = ""
#Loads the vectorstore created in data.py
def load_index():
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
  db = FAISS.load_local(
    "faiss_index", 
    embeddings,  
    allow_dangerous_deserialization=True #allow_dangerous_deserialization=True; this flag is set to allow loading locally stored indexes that may otherwise be malicious
  )

  return db

#Creating and configuring chain
def create_chain(llm, retriever):

  #Template to direct the agent
  rag_tempalte = """
  Answer the question based only on the following context:
  {context}
  Question: {question}
  you will also have access to chat history:
  {chat_history}
  """
  #Creating a prompt from the template 
  rag_prompt = ChatPromptTemplate.from_template(rag_tempalte)

  #The chain is configured to take inputs for chat history and question
  chain = (
      {
          "chat_history": itemgetter("chat_history"),
          "context": itemgetter("question") | retriever,#the question is chained to the retriever that then returns k relevant documents from the vector database
          "question": itemgetter("question")

      }
      | rag_prompt
      | llm
      | StrOutputParser()
  )

  return chain


def chatbot(chain):

  #List to store chat history
  ch = []
  while True:
    q = input("User :  ")
    if "exit" in q:
      break

    #Get user query and invoke chain with query and chat history
    res = chain.invoke({"chat_history": itemgetter(ch), "question": q})
    print("Llama : ", res)

    #Add current conversation to chat history
    ch.append((q, res))

def main():

  #loading the db and creating a retriever class
  db = load_index()
  retriever = db.as_retriever()

  #Initializing llama using groq api
  llm = ChatGroq(
    model_name = 'llama3-70b-8192'
    )
  chain = create_chain(llm,retriever)
  chatbot(chain)

if __name__=="__main__":
    main()
