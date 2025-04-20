import os
import uuid
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Internal Control PDF Helper API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize AWS clients
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

# Set up folder path for temporary files
folder_path = "./vector_store/"
os.makedirs(folder_path, exist_ok=True)

# Define models for request/response
class QueryRequest(BaseModel):
    request_id: str
    question: str

class QueryResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    request_id: str
    chunks: int

# Helper functions
def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(folder_path=folder_path, index_name=request_id)
    return request_id

# def create_vector_store(request_id, documents):
#     vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
#     file_name = f"{request_id}.bin"
#     vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

#     # Uncomment to use S3 storage
#     # s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key=f"{file_name}.faiss")
#     # s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key=f"{file_name}.pkl")

#     return file_name

def load_index(file_name):
    # Uncomment to use S3 storage
    # s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{file_name}.faiss", Filename=f"{folder_path}{file_name}.faiss")
    # s3_client.download_file(Bucket=BUCKET_NAME, Key=f"{file_name}.pkl", Filename=f"{folder_path}{file_name}.pkl")
    
    faiss_index = FAISS.load_local(
        index_name=file_name,
        folder_path=folder_path,
        embeddings=bedrock_embeddings
    )
    return faiss_index

def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client,
                   model_kwargs={'max_tokens_to_sample': 512})

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": question})
    return answer['result']

# API endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only TXT files are allowed")
    
    try:
        request_id = get_unique_id()
        save_file_name = f"{folder_path}{request_id}.txt"
        
        # Save the uploaded file
        with open(save_file_name, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the file
        loader = TextLoader(save_file_name, encoding='utf-8')
        pages = loader.load_and_split()
        splitted_docs = split_text(pages)

        # Create vector store
        file_name = create_vector_store(request_id, splitted_docs)

        return UploadResponse(
            request_id=request_id,
            chunks=len(splitted_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        file_name = f"{request.request_id}"
        vectorstore = load_index(file_name)
        llm = get_llm()
        
        response = get_response(llm, vectorstore, request.question)
        
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Internal Control PDF Helper API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 