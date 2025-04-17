from fastapi import FastAPI, File, UploadFile, Form
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage
import google.generativeai as genai

genai.configure(api_key="AIzaSyATNhxRcOI-VlwqNsCeOnW6vTcCCC4s_I4")
g_model = genai.GenerativeModel("gemini-2.0-flash-exp")

app = FastAPI()


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")
model = SentenceTransformer("all-MiniLM-L6-v2") 



@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_reader = PdfReader(file.file)
        text = "\n".join([page.extract_text()
                        for page in pdf_reader.pages if page.extract_text()])

        # Split text 
        text_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

        # Convert chunk to embeddings 
        for idx, chunk in enumerate(text_chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(ids=[f"{file.filename}_{idx}"],
                        embeddings=[embedding], documents=[chunk])

        return {"message": f"PDF '{file.filename}' uploaded and indexed successfully!"}
    except Exception as e:
        print(f"Error processing PDF: {e}")  
        return {"message": f"Error processing PDF: {str(e)}"}



def retrieve_docs(query):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    
    return results["documents"][0] if results["documents"] else "No relevant information found."


def generate_response(query):
    relevant_text = retrieve_docs(query)

    if relevant_text == "No relevant information found.":
        return relevant_text

    
    
    response = g_model.generate_content([f"Use this information to answer the users query provided in the next message: {relevant_text}", f"Query: {query}"])

    if response and hasattr(response, "candidates") and response.candidates:
        content = response.candidates[0].content  
        if content and hasattr(content, "parts") and content.parts:
            return content.parts[0].text  

    return "Could not generate a response."


@app.post("/chat/")
async def chat(user_query: str = Form(...)):
    response = generate_response(user_query)
    return {"answer": response}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot!"}


@app.post("/clear_database/")
async def clear_database():
    chroma_client.delete_collection(
        name="pdf_knowledge_base") 
    return {"message": "Database cleared successfully!"}

# Just testing PR Auto 
