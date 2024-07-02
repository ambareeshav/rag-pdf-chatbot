import PyPDF2
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = 'attention_is_all_you_need.pdf'
output_txt = 'attention_is_all_you_need.txt'

#Open the PDF file in read-binary mode
with open(pdf_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    #Initialize an empty string to store the text
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

#Write the extracted text to a text file and open it in read mode
with open(output_txt, 'w', encoding='utf-8') as txt_file:
    txt_file.write(text)

with open(output_txt, 'r', encoding='utf-8') as txt_file:
    text = txt_file.read()

#Function for counting tokens in txt file 
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

#Spliting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap  = 24,
    length_function = count_tokens,
)
chunks = text_splitter.split_text(text)

#Generate vector emdeddings for the text and store them in a FAISS vector database
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

db = FAISS.from_texts(texts=chunks,embedding=embeddings)

db.save_local("faiss_index")
print("Vector db saved locally!")