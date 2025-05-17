import os
import uuid
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv
from text_classifier import InternalControlClassifier
import pandas as pd
import openpyxl
from datetime import datetime
import tempfile
from google.cloud import vision
from google.cloud import storage
import json
import re

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

# Initialize GCP clients
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()
bucket = storage_client.get_bucket('twse-ocr-result')
feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

# Set up folder paths
folder_path = os.path.abspath("./vector_store/")
ocr_output_path = os.path.abspath("./ocr_output/")
os.makedirs(folder_path, exist_ok=True)
os.makedirs(ocr_output_path, exist_ok=True)

# Initialize the classifier
classifier = InternalControlClassifier()

# Define models for request/response
class QueryRequest(BaseModel):
    fileNames: list[str]

class QueryResponse(BaseModel):
    results: list[dict[str, str]]

class ClassificationRequest(BaseModel):
    fileNames: list[str]

class ClassificationResponse(BaseModel):
    results: list[dict[str, str | float]]

class UploadResponse(BaseModel):
    request_id: str
    chunks: int

class StoreResultRequest(BaseModel):
    results: list[dict[str, str]]

class StoreResultResponse(BaseModel):
    message: str
    file_path: str

class ExtractDateRequest(BaseModel):
    fileNames: list[str]

class ExtractDateResponse(BaseModel):
    results: list[dict[str, str]]

# Helper functions
def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    vectorstore_faiss.save_local(folder_path=folder_path, index_name=request_id)

    # if need to use S3 storage
    # s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key=f"{file_name}.faiss")
    # s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key=f"{file_name}.pkl")
    
    return request_id

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

def process_pdf_with_ocr(file_path: str, request_id: str, original_filename: str) -> str:
    """Process PDF file using GCP Vision OCR and return extracted text."""
    filename = os.path.basename(file_path)
    input_dir = os.path.dirname(file_path)
    
    # Upload to GCS
    remote_subdir = os.path.basename(os.path.normpath(input_dir))
    rel_remote_path = f"{remote_subdir}/{filename}"
    blob = bucket.blob(rel_remote_path)
    blob.upload_from_filename(file_path)

    # Configure OCR request
    gcs_source_uri = f'gs://twse-ocr-result/{rel_remote_path}'
    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type='application/pdf')

    gcs_destination_uri = f'gs://twse-ocr-result/{remote_subdir}/json_output/{filename[:30]}_'
    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

    # Perform OCR
    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config)
    operation = vision_client.async_batch_annotate_files(requests=[async_request])
    operation.result(timeout=180)

    # Get results
    prefix = '/'.join(gcs_destination_uri.split('//')[1].split('/')[1:])
    blobs_list = list(bucket.list_blobs(prefix=prefix))
    blobs_list = sorted(blobs_list, key=lambda blob: len(blob.name))

    # Extract text from results and save JSON files
    extracted_text = ''
    json_output_dir = os.path.join(ocr_output_path, request_id)
    os.makedirs(json_output_dir, exist_ok=True)

    for i, blob in enumerate(blobs_list):
        json_string = blob.download_as_string()
        response = json.loads(json_string)
        
        # Save JSON response
        json_file_path = os.path.join(json_output_dir, f'page_{i+1}.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        full_text_response = response['responses']
        for text in full_text_response:
            try:
                annotation = text['fullTextAnnotation']
                extracted_text += annotation['text']
            except:
                pass

    # Save extracted text with original filename
    text_filename = os.path.splitext(original_filename)[0] + '.txt'
    text_file_path = os.path.join(json_output_dir, text_filename)
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    return extracted_text

# API endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        request_id = get_unique_id()
        
        # Save uploaded PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Process PDF with OCR
        extracted_text = process_pdf_with_ocr(temp_file_path, request_id, file.filename)
        
        # Save extracted text to file for vector store with original filename
        text_filename = os.path.splitext(file.filename)[0] + '.txt'
        text_file_path = os.path.join(folder_path, text_filename)
        with open(text_file_path, "w", encoding='utf-8') as text_file:
            text_file.write(extracted_text)

        # Process the extracted text
        loader = TextLoader(text_file_path, encoding='utf-8')
        pages = loader.load_and_split()
        splitted_docs = split_text(pages)

        # Create vector store
        file_name = create_vector_store(request_id, splitted_docs)

        # Clean up temporary PDF file
        os.unlink(temp_file_path)

        return UploadResponse(
            request_id=request_id,
            chunks=len(splitted_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/queryBedrock", response_model=QueryResponse)
# async def query_bedrock(request: QueryRequest):
#     try:
#         results = []
#         question = "請用繁體中文回答，證券商被罰款多少錢？請以「新臺幣xx萬元」的格式回答"
#         
#         for filename in request.fileNames:
#             # Get the text file path
#             text_filename = os.path.splitext(filename)[0] + '.txt'
#             text_file_path = os.path.join(folder_path, text_filename)
#             
#             # Read the text content
#             with open(text_file_path, 'r', encoding='utf-8') as f:
#                 text_content = f.read()
#             
#             # Create a temporary vector store for this file
#             loader = TextLoader(text_file_path, encoding='utf-8')
#             pages = loader.load_and_split()
#             splitted_docs = split_text(pages)
#             vectorstore = FAISS.from_documents(splitted_docs, bedrock_embeddings)
#             
#             # Get LLM response
#             llm = get_llm()
#             response = get_response(llm, vectorstore, question)
#             
#             results.append({
#                 "fileName": filename,
#                 "response": response
#             })
#         
#         return QueryResponse(results=results)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/findFineAmount", response_model=QueryResponse)
async def find_fine_amount(request: QueryRequest):
    try:
        results = []
        # 正則表達式匹配 "新台幣" 或 "新臺幣" 後面跟著數字和"萬元"，但不包含"以上"
        pattern = r'新[台臺]幣(\d+)萬元(?!以上)'
        
        for filename in request.fileNames:
            # Get the text file path
            text_filename = os.path.splitext(filename)[0] + '.txt'
            text_file_path = os.path.join(folder_path, text_filename)
            
            # Read the text content
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # 移除所有空格
            text_content = text_content.replace(" ", "")
            
            # Find all matches
            matches = re.finditer(pattern, text_content)
            fine_amounts = []
            
            for match in matches:
                amount = match.group(0)  # 完整的匹配字串，如 "新台幣50萬元"
                fine_amounts.append(amount)
            
            # 如果找到多個金額，用逗號分隔
            response = "、".join(fine_amounts) if fine_amounts else "未找到罰款金額"
            
            results.append({
                "fileName": filename,
                "response": response
            })
        
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classifyText", response_model=ClassificationResponse)
async def classify_document(request: ClassificationRequest):
    try:
        results = []
        for filename in request.fileNames:
            # Get the text file path
            text_filename = os.path.splitext(filename)[0] + '.txt'
            text_file_path = os.path.join(folder_path, text_filename)
            
            # Read the text content
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Get document classification
            category, similarity_score = classifier.classify_document(text_content)
            
            results.append({
                "fileName": filename,
                "category": category,
                "similarity_score": float(similarity_score)
            })
        
        return ClassificationResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/storeResult", response_model=StoreResultResponse)
async def store_result(request: StoreResultRequest):
    try:
        # Create results directory if it doesn't exist
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Excel file path
        excel_path = os.path.join(results_dir, "internal_control_results.xlsx")
        
        # Create new rows data
        new_rows = []
        for result in request.results:
            new_row = {
                "Company Code": result["companyCode"],
                "Submitted Date": result["date"],
                "Fined Amount": result["finedAmount"],
                "Classification Category": result["category"],
                "Recorded Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            new_rows.append(new_row)
        
        # Check if file exists
        if os.path.exists(excel_path):
            # Read existing file
            df = pd.read_excel(excel_path)
        else:
            # Create new DataFrame with headers
            df = pd.DataFrame(columns=["Company Code", "Submitted Date", "Fined Amount", 
                                     "Classification Category", "Recorded Date"])
        
        # Append new rows
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Save to Excel
        df.to_excel(excel_path, index=False)
        
        return StoreResultResponse(
            message=f"Successfully stored {len(new_rows)} results",
            file_path=excel_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/downloadResults")
async def download_results():
    try:
        excel_path = "./results/internal_control_results.xlsx"
        if not os.path.exists(excel_path):
            raise HTTPException(status_code=404, detail="No results file found")
        
        return FileResponse(
            path=excel_path,
            filename="internal_control_results.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extractDate", response_model=ExtractDateResponse)
async def extract_date(request: ExtractDateRequest):
    try:
        results = []
        for filename in request.fileNames:
            # Get the text file path
            text_filename = os.path.splitext(filename)[0] + '.txt'
            text_file_path = os.path.join(folder_path, text_filename)
            
            # Read the text content
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Remove all spaces from the text before matching
            text_without_spaces = text_content.replace(" ", "")
            
            # Pattern to match date after "日期:"
            pattern = r'日期:(\d{3}年\d{1,2}月\d{1,2}日)'
            # Using re.search() to get the first match after "日期:"
            match = re.search(pattern, text_without_spaces)
            
            if match:
                # Return the captured date group
                results.append({
                    "fileName": filename,
                    "date": match.group(1)
                })
            else:
                results.append({
                    "fileName": filename,
                    "date": "No date found"
                })
            
        return ExtractDateResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Internal Control PDF Helper API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 