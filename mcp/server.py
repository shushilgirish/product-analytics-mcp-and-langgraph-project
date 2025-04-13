from fastapi import FastAPI
from pydantic import BaseModel
import os
from pinecone import Pinecone
from openai import OpenAI
import boto3
import json
import uvicorn

app = FastAPI()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
s3_client = boto3.client('s3')

class Query(BaseModel):
    query: str

@app.post("/generate_summary")
async def generate_summary(query: Query):
    """
    Generate a summary based on the query using vector search and stored chunks.
    """
    # Get query embedding
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query.query]
    )
    query_vector = response.data[0].embedding
    
    # Search Pinecone
    index = pc.Index("healthcare-product-analytics")
    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        namespace="book-kotler"
    )
    
    # Get relevant chunks from S3
    context = []
    for match in search_results['matches']:
        if 's3_chunks_key' in match['metadata']:
            s3_key = match['metadata']['s3_chunks_key']
            response = s3_client.get_object(
                Bucket="finalproject-product",
                Key=s3_key
            )
            chunks_data = json.loads(response['Body'].read().decode('utf-8'))
            chunk_id = match['id']
            if chunk_id in chunks_data['chunks']:
                context.append(chunks_data['chunks'][chunk_id]['text'])
    
    # Generate summary
    if context:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear and concise summaries."},
                {"role": "user", "content": f"Based on this information: {' '.join(context)}\n\nAnswer this question: {query.query}"}
            ],
            max_tokens=300
        )
        return {"summary": response.choices[0].message.content}
    else:
        return {"summary": "No relevant information found."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
