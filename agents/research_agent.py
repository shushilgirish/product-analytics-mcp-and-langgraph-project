import os
import sys
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import requests
import httpx
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body, BackgroundTasks
from fastapi.responses import JSONResponse

# Add paths for shared libraries if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("research_agent.log")
    ]
)
logger = logging.getLogger("research_agent")

# Load configuration
try:
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "Airflow/config/book.json"), "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logger.warning("Book configuration not found, using defaults")
    CONFIG = {
        "PINECONE_INDEX_NAME": "healthcare-product-analytics",
        "EMBEDDING_DIMENSION": 1536
    }

# Initialize FastAPI app
app = FastAPI(
    title="MarketScope Research Agent",
    description="Knowledge retrieval agent for MarketScope AI Platform",
    version="0.1.0"
)

# Agent configuration
AGENT_CONFIG = {
    "name": "Research Agent",
    "description": "Retrieves information from knowledge bases and performs research",
    "capabilities": ["query_knowledge", "extract_insights", "summarize_content"],
    "endpoint_url": os.getenv("AGENT_ENDPOINT", "http://localhost:8001/invoke"),
    "mcp_url": os.getenv("MCP_URL", "http://localhost:8000"),
    "api_key": os.getenv("MCP_API_KEY", "development_key"),
    "pinecone_index": CONFIG.get("PINECONE_INDEX_NAME", "healthcare-product-analytics"),
    "pinecone_namespace": "book-kotler"  # Namespace where your book content is stored
}

# Agent state
agent_state = {
    "registered": False,
    "agent_id": None,
    "active_sessions": {},
    "tasks_processed": 0,
    "last_heartbeat": None
}

# Pydantic models
class AgentRequest(BaseModel):
    task_id: str
    action: str
    parameters: Dict[str, Any] = {}
    session_id: Optional[str] = None
    context: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    status: str = "success"
    result: Dict[str, Any] = {}
    error: Optional[str] = None
    task_id: Optional[str] = None
    execution_time: Optional[float] = None

# Register with MCP on startup
@app.on_event("startup")
async def startup_event():
    try:
        await register_with_mcp()
        # Start heartbeat
        background_tasks = BackgroundTasks()
        background_tasks.add_task(heartbeat_loop)
    except Exception as e:
        logger.error(f"Failed to register with MCP: {e}")

async def register_with_mcp():
    """Register the agent with the MCP server"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{AGENT_CONFIG['mcp_url']}/agents/register",
                json={
                    "name": AGENT_CONFIG["name"],
                    "description": AGENT_CONFIG["description"],
                    "capabilities": AGENT_CONFIG["capabilities"],
                    "endpoint_url": AGENT_CONFIG["endpoint_url"]
                },
                headers={"api-key": AGENT_CONFIG["api_key"]}
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_state["registered"] = True
                agent_state["agent_id"] = result["agent_id"]
                agent_state["last_heartbeat"] = datetime.now()
                logger.info(f"Successfully registered with MCP as {result['agent_id']}")
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error registering with MCP: {str(e)}")
            raise

async def send_heartbeat():
    """Send heartbeat to MCP"""
    if not agent_state["registered"] or not agent_state["agent_id"]:
        logger.warning("Cannot send heartbeat - agent not registered")
        return False
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{AGENT_CONFIG['mcp_url']}/agents/heartbeat",
                json={"agent_id": agent_state["agent_id"]},
                headers={"api-key": AGENT_CONFIG["api_key"]}
            )
            
            if response.status_code == 200:
                agent_state["last_heartbeat"] = datetime.now()
                return True
            else:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending heartbeat: {str(e)}")
            return False

async def heartbeat_loop():
    """Continuous heartbeat loop"""
    while True:
        try:
            await send_heartbeat()
        except Exception as e:
            logger.error(f"Heartbeat error: {str(e)}")
            
        # Wait 60 seconds before next heartbeat
        await asyncio.sleep(60)

# Main agent endpoint
@app.post("/invoke")
async def invoke(request: AgentRequest):
    """Process incoming requests from MCP"""
    start_time = time.time()
    logger.info(f"Received {request.action} request: {request.task_id}")
    
    try:
        # Process based on the requested action
        if request.action == "query_knowledge":
            result = await query_knowledge(request.parameters)
        elif request.action == "extract_insights":
            result = await extract_insights(request.parameters)
        elif request.action == "summarize_content":
            result = await summarize_content(request.parameters)
        else:
            return AgentResponse(
                status="error",
                error=f"Unsupported action: {request.action}",
                task_id=request.task_id,
                execution_time=time.time() - start_time
            )
        
        # Track task processing
        agent_state["tasks_processed"] += 1
        
        # Report task completion
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            report_task_completion,
            task_id=request.task_id,
            result=result
        )
        
        return AgentResponse(
            status="success",
            result=result,
            task_id=request.task_id,
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return AgentResponse(
            status="error",
            error=f"Internal agent error: {str(e)}",
            task_id=request.task_id,
            execution_time=time.time() - start_time
        )

async def report_task_completion(task_id: str, result: Dict):
    """Report task completion to MCP"""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{AGENT_CONFIG['mcp_url']}/tasks/{task_id}/callback",
                json=result,
                headers={"api-key": AGENT_CONFIG["api_key"]}
            )
        except Exception as e:
            logger.error(f"Error reporting task completion: {str(e)}")

# Knowledge actions
async def query_knowledge(parameters: Dict) -> Dict:
    """Query the knowledge base using Pinecone"""
    query = parameters.get("query")
    if not query:
        raise ValueError("Query parameter is required")
    
    max_results = parameters.get("max_results", 5)
    
    # Generate embedding for the query
    embedding = await generate_embedding(query)
    
    # Query Pinecone
    results = await query_pinecone(embedding, max_results)
    
    # Process and format results
    processed_results = []
    for match in results.get("matches", []):
        processed_results.append({
            "id": match.get("id", "unknown"),
            "score": match.get("score", 0),
            "content": match.get("metadata", {}).get("text_preview", "No preview available"),
            "source": match.get("metadata", {}).get("source", "Unknown source"),
            "chunk_idx": match.get("metadata", {}).get("chunk_idx", -1)
        })
    
    return {
        "query": query,
        "results": processed_results,
        "result_count": len(processed_results)
    }

async def extract_insights(parameters: Dict) -> Dict:
    """Extract insights from text content"""
    text = parameters.get("text", "")
    if not text:
        raise ValueError("Text parameter is required")
    
    # In a real implementation, this would use an LLM to extract insights
    # For now, we'll return a placeholder
    return {
        "insights": [
            "Placeholder insight 1",
            "Placeholder insight 2",
            "Placeholder insight 3"
        ],
        "keywords": ["marketing", "analysis", "strategy"],
        "sentiment": "positive"
    }

async def summarize_content(parameters: Dict) -> Dict:
    """Summarize content"""
    text = parameters.get("text", "")
    if not text:
        raise ValueError("Text parameter is required")
    
    # In a real implementation, this would use an LLM to generate a summary
    # For now, we'll return a placeholder
    return {
        "summary": "This is a placeholder summary of the provided content.",
        "length": len(text),
        "topics": ["marketing", "strategy", "analysis"]
    }

# Pinecone integration
async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI API"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return a dummy embedding (in production, handle this better)
        return [0.0] * 1536

async def query_pinecone(embedding: List[float], top_k: int = 5) -> Dict:
    """Query Pinecone index with the given embedding"""
    try:
        from pinecone import Pinecone
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Access the index
        index = pc.Index(AGENT_CONFIG["pinecone_index"])
        
        # Query the index
        query_response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=AGENT_CONFIG["pinecone_namespace"]
        )
        
        return query_response
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return {"matches": []}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Agent health check"""
    return {
        "status": "online",
        "agent": AGENT_CONFIG["name"],
        "registered": agent_state["registered"],
        "agent_id": agent_state["agent_id"],
        "tasks_processed": agent_state["tasks_processed"],
        "last_heartbeat": agent_state["last_heartbeat"].isoformat() if agent_state["last_heartbeat"] else None,
        "pinecone_index": AGENT_CONFIG["pinecone_index"],
        "pinecone_namespace": AGENT_CONFIG["pinecone_namespace"]
    }

# Run server
if __name__ == "__main__":
    port = int(os.getenv("AGENT_PORT", "8001"))
    logger.info(f"Starting Research Agent on port {port}")
    uvicorn.run("research_agent:app", host="0.0.0.0", port=port, reload=True)