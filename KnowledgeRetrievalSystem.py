import sys
import certifi
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import duckdb
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uvicorn
import logging
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize DuckDB and FAISS index
def initialize_db():
    try:
        conn = duckdb.connect('knowledge.db')
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_knowledgeid START 1;")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            knowledgeid INTEGER DEFAULT nextval('seq_knowledgeid'),
            content TEXT
        );
        """)
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing the database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing the database: {str(e)}")

try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Initialize FAISS index (initially set to None)
index = None
document_embeddings = []
documents = []

# Initialize the database and FAISS index
initialize_db()

# Pydantic model for knowledge (document)
class Knowledge(BaseModel):
    content: str

# 1. To Get Knowledge from DuckDB
@app.get("/get-knowledge")
def get_knowledge():
    try:
        conn = duckdb.connect('knowledge.db')
        result = conn.execute("SELECT content FROM knowledge").fetchall()
        conn.close()
        logger.info(f"Fetched {len(result)} rows from knowledge.")
        return {"knowledge": result}
    except Exception as e:
        logger.error(f"Error fetching knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching knowledge: {str(e)}")


# 2. upload to DuckDB and start training
@app.post("/addKnowledge")
def addKnowledge(new_knowledge: list[Knowledge]):
    global documents, document_embeddings, index
    try:
        # List to store the chunks of text
        chunks = []
        for knowledge_item in new_knowledge:
            # Split by sentences for better context
            knowledge_chunks = sent_tokenize(knowledge_item.content)  # Split the content into sentences
            chunks.extend(knowledge_chunks)  # Flatten the list of chunks

        documents = chunks  # Now the documents are the individual chunks (sentences)
        document_embeddings = model.encode(documents)  # Get embeddings for each chunk

        if len(document_embeddings) == 0:
            raise HTTPException(status_code=400, detail="No embeddings found to add to FAISS index.")

        # Get the embedding dimension dynamically (for flexibility in case of model change)
        embedding_dim = document_embeddings[0].shape[0]  # Use the dimension of the first embedding

        # Initialize FAISS index with the dynamic dimension
        if index is None:  # Initialize FAISS index only once
            index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"FAISS index initialized with dimension: {embedding_dim}")

        # Convert to numpy array and ensure correct shape
        embeddings_array = np.array(document_embeddings).astype(np.float32)

        # Add embeddings to FAISS
        index.add(embeddings_array)

        # Store the chunks into DuckDB
        conn = duckdb.connect('knowledge.db')
        for chunk in chunks:
            conn.execute("INSERT INTO knowledge (content) VALUES (?)", (chunk,))
        conn.commit()
        conn.close()

        logger.info("Training started with provided knowledge!")
        return {"message": "Training started with provided knowledge!"}
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
    

# 3. Refresh Training (clear existing training and train new)
@app.post("/refresh-training")
def refresh_training():
    global documents, document_embeddings
    try:
        # Clear existing embeddings in FAISS
        index.reset()

        # Re-fetch and re-train with data from DuckDB
        conn = duckdb.connect('knowledge.db')
        knowledge_data = conn.execute("SELECT content FROM knowledge").fetchall()
        conn.close()

        documents = [row[0] for row in knowledge_data]
        document_embeddings = model.encode(documents)

        # Re-add embeddings to FAISS
        embeddings_array = np.array(document_embeddings).astype(np.float32)
        index.add(embeddings_array)

        logger.info("Training refreshed with new knowledge!")
        return {"message": "Training refreshed with new knowledge!"}
    except Exception as e:
        logger.error(f"Error during refresh training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during refresh training: {str(e)}")

# 4. Update Training (upload new document and update training)
@app.post("/update-training")
def update_training(new_knowledge: Knowledge):
    global documents, document_embeddings
    try:
        # Insert new knowledge into DuckDB
        conn = duckdb.connect('knowledge.db')
        conn.execute("INSERT INTO knowledge (content) VALUES (?)", (new_knowledge.content,))
        conn.commit()
        conn.close()

        # Re-fetch and re-train with updated data
        conn = duckdb.connect('knowledge.db')
        knowledge_data = conn.execute("SELECT content FROM knowledge").fetchall()
        conn.close()

        documents = [row[0] for row in knowledge_data]
        document_embeddings = model.encode(documents)

        # Re-add embeddings to FAISS
        embeddings_array = np.array(document_embeddings).astype(np.float32)
        index.add(embeddings_array)

        logger.info("Training updated with new knowledge!")
        return {"message": "Training updated with new knowledge!"}
    except Exception as e:
        logger.error(f"Error updating training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating training: {str(e)}")


# 5. Ask Questions (Using FAISS for Retrieval and Generative Model for Answer)
@app.get("/ask")
def ask_question(query: str):
    try:
        answer = generate_answer_from_query(query)
        logger.info(f"Answer generated: {answer}")
        return {"answer": answer}  # Return the generated answer

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
    
# Function to load FAISS index from DuckDB if not already loaded
def load_index_from_db():
    global index, documents
    try:
        # Connect to DuckDB to fetch the data
        conn = duckdb.connect('knowledge.db')
        knowledge_data = conn.execute("SELECT content FROM knowledge").fetchall()
        conn.close()

        # Extract text from DuckDB result
        documents = [row[0] for row in knowledge_data]

        # Generate embeddings for documents and initialize FAISS index
        document_embeddings = model.encode(documents)
        embedding_dim = document_embeddings[0].shape[0]  # Get the embedding dimension dynamically

        # Initialize FAISS index if it doesn't exist
        if index is None:
            index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"FAISS index initialized with dimension: {embedding_dim}")

        # Convert embeddings to numpy array
        embeddings_array = np.array(document_embeddings).astype(np.float32)
        
        # Add embeddings to FAISS index
        index.add(embeddings_array)
        logger.info("Loaded documents and added embeddings to FAISS index.")
        
    except Exception as e:
        logger.error(f"Error loading FAISS index from DB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading FAISS index from DB: {str(e)}")

# Reusable function to handle the query and fetch answer
def generate_answer_from_query(query: str):
    try:
        query_embedding = model.encode([query]).astype(np.float32)
        
        # If index is not loaded, load it from DB
        if index is None:
            load_index_from_db()

        # Perform the similarity search with FAISS to retrieve the most relevant chunks
        k = 10  # Number of relevant chunks to retrieve
        _, indices = index.search(query_embedding, k)

        # Retrieve the most relevant chunk(s)
        relevant_docs = [documents[i] for i in indices[0]]


       # Remove very similar documents based on cosine similarity
        filtered_docs = []
        for i in range(len(relevant_docs)):
            # Compare with previously added documents
            add = True
            for doc in filtered_docs:
                # Calculate cosine similarity
                sim = cosine_similarity([model.encode([relevant_docs[i]])[0]], [model.encode([doc])][0])
                if sim > 0.95:  # Adjust the threshold as needed
                    add = False
                    break
            if add:
                filtered_docs.append(relevant_docs[i])

        # Combine the filtered documents for context
        context = "\n".join(filtered_docs)  # Combine the top k relevant chunks


        # Use a transformer model (BART) for generating an answer
        generator = pipeline("text-generation", model="facebook/bart-large-cnn")
        answer = generator(f"Answer the following question based on the text: {context}", max_length=200)

        logger.info(f"Answer generated: {answer[0]['generated_text']}")
        return answer[0]['generated_text']
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# 6. Root Endpoint (Added root() for basic API health check)
@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application!"}

# Main entry point to run the application
def main():
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
