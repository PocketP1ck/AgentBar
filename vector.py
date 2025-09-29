from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the AI agents dataset
df = pd.read_csv("data.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./ai_agents_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        # Create a comprehensive document content for better search
        page_content = f"""
        Name: {row['name']}
        Category: {row['category']}
        Description: {row['description']}
        Links: {row['links']}
        """
        
        document = Document(
            page_content=page_content,
            metadata={
                "name": row['name'],
                "category": row['category'],
                "description": row['description'],
                "links": row['links']
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="ai_agents",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Return top 5 most relevant agents
)
