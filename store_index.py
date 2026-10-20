import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


extracted_docs = load_pdf_files("data/")
filter_data = filter_to_minimal_docs(extracted_docs)
text_chunks = text_split(filter_data)

embedding = embeddings



pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)


index_name = 'medai'

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,       #dimension of the embedding
        metric = 'cosine',     #metric for similarity search
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )


index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embedding,
    index_name = index_name,
)
