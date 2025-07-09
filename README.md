# Lexi Legal RAG Backend

## Setup Instructions

### 1. Clone the Repository

```bash
# Clone this repository and navigate to the project directory
cd lexi.sg-rag-backend-test
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# If you haven't already, also install:
pip install flask sentence-transformers faiss-cpu numpy python-docx pymupdf transformers accelerate torch
```

### 4. Prepare the Document Index

- Place your legal PDF and DOCX files in the `sample_docs/` directory.
- Run the index builder to create the FAISS index and metadata:

```bash
python rag_index_builder.py
```

### 5. Start the Flask Server

```bash
python app/server.py
```

The server will start at `http://127.0.0.1:5000`.

---

## How to Test the API

### Using Python Test Script

Run the provided test script:

```bash
python app/test_server.py
```

### Using curl

```bash
curl -X POST http://127.0.0.1:5000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"}'
```

### Using Postman

1. Open Postman and create a new POST request to `http://127.0.0.1:5000/query`
2. Set the header `Content-Type: application/json`
3. In the Body (raw, JSON), enter your query:
   ```json
   {
     "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
   }
   ```
4. Click Send and view the response.

---

## Example Input/Output

### Example 1

screenshots/Screenshot 2025-07-09 173049.png
screenshots/Screenshot 2025-07-09 173033.png
screenshots/Screenshot 2025-07-09 172936.png

### Example 2

screenshots/Screenshot 2025-07-09 174224.png
screenshots/Screenshot 2025-07-09 174206.png
screenshots/Screenshot 2025-07-09 174144.png

---

## Notes

- The backend uses a local, open-source LLM (Phi-3 Mini) for answer generation.
- All document embeddings and retrieval are performed locally using FAISS and sentence-transformers.
- For best results, ensure your `sample_docs/` folder contains relevant legal documents in PDF or DOCX format.
