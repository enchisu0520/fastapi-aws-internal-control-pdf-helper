# Internal Control PDF Helper Backend

This is the backend service for the Internal Control PDF Helper application. It uses FastAPI and AWS Bedrock to process text files and provide AI-powered responses to queries.

## Prerequisites

- Python 3.8 or higher
- AWS account with Bedrock access
- AWS credentials with appropriate permissions

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: source venv/Scripts/activate
```

2. Install dependencies:
```bash
conda install swig
pip install -r requirements.txt
```

### Windows-Specific Installation Notes

If you encounter issues installing `faiss-cpu` on Windows, follow these steps:

1. Install SWIG (required for building faiss-cpu):
   - Download SWIG from [http://www.swig.org/download.html](http://www.swig.org/download.html)
   - Extract the zip file
   - Add the extracted directory to your PATH environment variable
   - Restart your terminal and try installing again

2. Alternatively, use a pre-built wheel:
   ```bash
   pip install faiss-cpu==1.7.4 --find-links https://download.pytorch.org/whl/torch_stable.html
   ```
   or use anaconda
  ```bash
  conda install swig
  ```

3. Configure AWS credentials:
   - Copy the `.env.example` file to `.env`
   - Update the `.env` file with your AWS credentials:
     ```
     AWS_ACCESS_KEY_ID=your_access_key_id
     AWS_SECRET_ACCESS_KEY=your_secret_access_key
     AWS_REGION=us-east-1
     ```

## Running the Server

Start the FastAPI server:
```bash
python main.py
```

The server will run on `http://localhost:8000`

## API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Upload File
- **POST** `/upload`
- Accepts a text file upload
- Returns a request ID and number of chunks processed

### Query
- **POST** `/query`
- Request body:
  ```json
  {
    "request_id": "uuid_from_upload",
    "question": "your question here"
  }
  ```
- Returns AI-generated response based on the uploaded document

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (missing file, invalid file type, etc.)
- 500: Server Error (processing errors)

## Security Notes

- Never commit your `.env` file with real credentials
- Ensure proper AWS IAM permissions are set up
- Use HTTPS in production 