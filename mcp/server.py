import os
import uuid
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import asyncio
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_server.log")
    ]
)
logger = logging.getLogger("mcp_server")

# Load configuration
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logger.warning("Config file not found, using defaults")
    CONFIG = {
        "pinecone_index": "healthcare-product-analytics",
        "pinecone_namespace": "book-kotler",
        "session_timeout_hours": 24,
        "agent_timeout_minutes": 5
    }

# Initialize FastAPI app
app = FastAPI(
    title="MarketScope MCP Server",
    description="Master Control Program for MarketScope AI Platform",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API interactions
class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    endpoint_url: str
    api_key: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Research Agent",
                "description": "Retrieves and analyzes information from knowledge bases",
                "capabilities": ["query_knowledge", "analyze_text", "summarize"],
                "endpoint_url": "http://localhost:8001/invoke"
            }
        }

class AgentRequest(BaseModel):
    agent_id: str
    action: str
    parameters: Dict[str, Any] = {}
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "research-agent-12345678",
                "action": "query_knowledge",
                "parameters": {
                    "query": "What are the key marketing strategies for healthcare products?",
                    "max_results": 5
                }
            }
        }

class TaskCreate(BaseModel):
    task_type: str
    priority: int = 1
    parameters: Dict[str, Any] = {}
    callback_url: Optional[str] = None
    agent_preference: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "task_type": "market_analysis",
                "priority": 2,
                "parameters": {
                    "industry": "healthcare",
                    "focus": "product positioning"
                }
            }
        }

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    max_results: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "query": "How should healthcare products be marketed to hospitals?",
                "max_results": 3
            }
        }

class MCP:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.agent_registry: Dict[str, Dict] = {}
        self.task_queue: Dict[str, Dict] = {}
        self.results_cache: Dict[str, Dict] = {}
        
        # Start background tasks
        self.start_background_tasks()
    
    def register_agent(self, agent_id: str, agent_info: Dict) -> Dict:
        """Register a new agent with the MCP"""
        if agent_id in self.agent_registry:
            # Update existing agent info
            self.agent_registry[agent_id].update({
                **agent_info,
                "last_heartbeat": datetime.now()
            })
            logger.info(f"Updated agent registration: {agent_id}")
        else:
            # Register new agent
            self.agent_registry[agent_id] = {
                **agent_info,
                "registered_at": datetime.now(),
                "status": "online",
                "last_heartbeat": datetime.now(),
                "tasks_processed": 0
            }
            logger.info(f"New agent registered: {agent_id}")
            
        return self.agent_registry[agent_id]
    
    def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat time"""
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["last_heartbeat"] = datetime.now()
            return True
        return False
    
    def get_agent_by_capability(self, capability: str) -> Optional[str]:
        """Find an agent with the specified capability"""
        for agent_id, info in self.agent_registry.items():
            if info["status"] == "online" and capability in info.get("capabilities", []):
                return agent_id
        return None
    
    def create_session(self) -> str:
        """Create a new client session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "history": [],
            "context": {},
            "status": "active"
        }
        return session_id
    
    def update_session(self, session_id: str) -> bool:
        """Update session last active time"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_active"] = datetime.now()
            return True
        return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def add_to_session_history(self, session_id: str, entry: Dict) -> bool:
        """Add an entry to the session history"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                **entry,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False
    
    def add_task(self, task_info: Dict) -> str:
        """Add a task to the queue"""
        task_id = task_info.get("task_id", str(uuid.uuid4()))
        self.task_queue[task_id] = {
            **task_info,
            "created_at": datetime.now(),
            "status": "pending",
            "attempts": 0
        }
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information"""
        return self.task_queue.get(task_id)
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict] = None) -> bool:
        """Update task status and optionally add result"""
        if task_id in self.task_queue:
            self.task_queue[task_id]["status"] = status
            self.task_queue[task_id]["updated_at"] = datetime.now()
            
            if result:
                self.task_queue[task_id]["result"] = result
                
            return True
        return False
    
    def add_to_cache(self, key: str, data: Dict, ttl_seconds: int = 3600) -> None:
        """Add data to results cache with TTL"""
        self.results_cache[key] = {
            "data": data,
            "expires_at": datetime.now() + timedelta(seconds=ttl_seconds)
        }
    
    def get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from cache if not expired"""
        if key in self.results_cache:
            cache_item = self.results_cache[key]
            if datetime.now() < cache_item["expires_at"]:
                return cache_item["data"]
            else:
                # Remove expired item
                del self.results_cache[key]
        return None
    
    def start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # In production, you'd use a proper background task scheduler
        # For now, we'll just log that this would happen
        logger.info("Background tasks would start here (session cleanup, task processing)")

# Initialize MCP
mcp = MCP()

# Authentication dependency
async def get_api_key(api_key: str = Header(..., description="API Key for authentication")):
    # In production, implement proper API key validation
    valid_key = os.getenv("MCP_API_KEY", "development_key")
    if api_key != valid_key:
        logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Optional session dependency
async def get_session(session_id: Optional[str] = Header(None, description="Session ID for stateful operations")):
    if session_id:
        session = mcp.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        mcp.update_session(session_id)
        return session_id
    return None

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online", 
        "service": "MarketScope MCP Server", 
        "time": datetime.now().isoformat(),
        "agents_registered": len(mcp.agent_registry),
        "active_sessions": len(mcp.sessions)
    }

@app.post("/agents/register", response_model=Dict)
async def register_agent(agent_info: AgentInfo, api_key: str = Depends(get_api_key)):
    """Register an agent with the MCP"""
    agent_id = f"{agent_info.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
    registered = mcp.register_agent(agent_id, agent_info.dict())
    return {"agent_id": agent_id, "registered": registered}

@app.post("/agents/heartbeat")
async def agent_heartbeat(
    agent_id: str = Body(..., embed=True), 
    status: str = Body("online", embed=True),
    api_key: str = Depends(get_api_key)
):
    """Update agent heartbeat"""
    if not mcp.update_agent_heartbeat(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Update agent status if provided
    if status and agent_id in mcp.agent_registry:
        mcp.agent_registry[agent_id]["status"] = status
        
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/agents", response_model=Dict)
async def list_agents(api_key: str = Depends(get_api_key)):
    """List all registered agents"""
    # Filter out sensitive info like API keys
    safe_registry = {}
    for agent_id, info in mcp.agent_registry.items():
        safe_registry[agent_id] = {
            k: v for k, v in info.items() if k != "api_key"
        }
    
    return {
        "agents": safe_registry,
        "total": len(safe_registry),
        "online": sum(1 for a in safe_registry.values() if a.get("status") == "online")
    }

@app.post("/agents/{agent_id}/invoke")
async def invoke_agent(
    agent_id: str, 
    request: AgentRequest, 
    background_tasks: BackgroundTasks,
    session: Optional[str] = Depends(get_session),
    api_key: str = Depends(get_api_key)
):
    """Invoke an agent to perform an action"""
    # Check if agent exists
    if agent_id not in mcp.agent_registry:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = mcp.agent_registry[agent_id]
    
    # Check if agent is online
    if agent.get("status") != "online":
        raise HTTPException(status_code=503, detail=f"Agent {agent_id} is currently offline")
    
    # Create or use existing session
    session_id = session or request.session_id or mcp.create_session()
    
    # Update session
    if not mcp.update_session(session_id):
        # Create session if it doesn't exist
        session_id = mcp.create_session()
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Prepare request for agent
    agent_payload = {
        "task_id": task_id,
        "action": request.action,
        "parameters": request.parameters,
        "session_id": session_id,
        "context": request.context or {}
    }
    
    # Add task to queue
    mcp.add_task({
        "task_id": task_id,
        "agent_id": agent_id,
        "action": request.action,
        "session_id": session_id
    })
    
    # Add to session history
    if session_id:
        mcp.add_to_session_history(session_id, {
            "type": "agent_request",
            "agent_id": agent_id,
            "action": request.action,
            "task_id": task_id
        })
    
    # Send request to agent (async)
    background_tasks.add_task(
        call_agent_async, 
        agent_endpoint=agent["endpoint_url"],
        payload=agent_payload,
        task_id=task_id,
        session_id=session_id
    )
    
    return {
        "status": "request_accepted",
        "session_id": session_id,
        "task_id": task_id,
        "message": f"Request sent to {agent_id}"
    }

@app.post("/query")
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    session: Optional[str] = Depends(get_session),
    api_key: str = Depends(get_api_key)
):
    """Query the knowledge base using the research agent"""
    # Find a research agent
    agent_id = mcp.get_agent_by_capability("query_knowledge")
    if not agent_id:
        raise HTTPException(status_code=503, detail="No research agent available")
    
    # Create or use existing session
    session_id = session or request.session_id or mcp.create_session()
    
    # Create task
    agent_request = AgentRequest(
        agent_id=agent_id,
        action="query_knowledge",
        parameters={
            "query": request.query,
            "max_results": request.max_results
        },
        session_id=session_id,
        context=request.context
    )
    
    # Invoke agent
    return await invoke_agent(
        agent_id=agent_id,
        request=agent_request,
        background_tasks=background_tasks,
        session=session_id,
        api_key=api_key
    )

@app.post("/sessions/create")
async def create_session(api_key: str = Depends(get_api_key)):
    """Create a new client session"""
    session_id = mcp.create_session()
    return {"session_id": session_id, "created_at": datetime.now().isoformat()}

@app.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    include_history: bool = Query(False, description="Include session history"),
    api_key: str = Depends(get_api_key)
):
    """Get session information"""
    session = mcp.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Don't include history unless explicitly requested
    if not include_history:
        session = {k: v for k, v in session.items() if k != "history"}
    
    return {
        "session_id": session_id,
        "session": session
    }

@app.post("/tasks")
async def create_task(
    task: TaskCreate, 
    background_tasks: BackgroundTasks,
    session: Optional[str] = Depends(get_session),
    api_key: str = Depends(get_api_key)
):
    """Create a new task"""
    # Find an agent for this task
    agent_id = task.agent_preference
    if not agent_id:
        # Find an agent with the required capability
        agent_id = mcp.get_agent_by_capability(task.task_type)
    
    if not agent_id:
        raise HTTPException(status_code=503, detail=f"No agent available for task type: {task.task_type}")
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Add task to queue
    mcp.add_task({
        "task_id": task_id,
        "agent_id": agent_id,
        "type": task.task_type,
        "priority": task.priority,
        "parameters": task.parameters,
        "callback_url": task.callback_url,
        "session_id": session
    })
    
    # Process task in background
    background_tasks.add_task(
        process_task,
        task_id=task_id
    )
    
    return {"task_id": task_id, "status": "accepted"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, api_key: str = Depends(get_api_key)):
    """Get task status"""
    task = mcp.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "created_at": task.get("created_at").isoformat(),
        "result": task.get("result")
    }

@app.post("/tasks/{task_id}/callback")
async def task_callback(
    task_id: str, 
    result: Dict = Body(...), 
    api_key: str = Depends(get_api_key)
):
    """Callback for agent task completion"""
    task = mcp.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task status
    mcp.update_task_status(task_id, "completed", result)
    
    # Add to session history if applicable
    session_id = task.get("session_id")
    if session_id:
        mcp.add_to_session_history(session_id, {
            "type": "task_completed",
            "task_id": task_id,
            "result": result
        })
    
    # Handle callback URL if specified
    callback_url = task.get("callback_url")
    if callback_url:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json={
                    "task_id": task_id,
                    "status": "completed",
                    "result": result
                })
        except Exception as e:
            logger.error(f"Error sending callback to {callback_url}: {str(e)}")
    
    return {"status": "ok"}

# Helper functions
async def call_agent_async(agent_endpoint: str, payload: Dict, task_id: str, session_id: Optional[str] = None):
    """Call an agent asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent_endpoint, 
                json=payload, 
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update task status
                mcp.update_task_status(task_id, "completed", result)
                
                # Add result to session history
                if session_id:
                    mcp.add_to_session_history(session_id, {
                        "type": "agent_response",
                        "task_id": task_id,
                        "result": result
                    })
                
                logger.info(f"Agent call successful: {task_id}")
                return result
            else:
                error = f"Agent returned status {response.status_code}: {response.text}"
                logger.error(f"Agent call failed: {error}")
                
                # Update task status
                mcp.update_task_status(task_id, "failed", {"error": error})
                
                # Add error to session history
                if session_id:
                    mcp.add_to_session_history(session_id, {
                        "type": "agent_error",
                        "task_id": task_id,
                        "error": error
                    })
                
                return {"error": error}
                
    except Exception as e:
        error = f"Error calling agent: {str(e)}"
        logger.error(error)
        
        # Update task status
        mcp.update_task_status(task_id, "failed", {"error": error})
        
        # Add error to session history
        if session_id:
            mcp.add_to_session_history(session_id, {
                "type": "agent_error",
                "task_id": task_id,
                "error": error
            })
        
        return {"error": error}

async def process_task(task_id: str):
    """Process a task from the queue"""
    task = mcp.get_task(task_id)
    if not task:
        logger.error(f"Task not found: {task_id}")
        return
    
    agent_id = task.get("agent_id")
    if not agent_id:
        logger.error(f"No agent assigned for task: {task_id}")
        mcp.update_task_status(task_id, "failed", {"error": "No agent assigned"})
        return
    
    # Get agent info
    agent = mcp.agent_registry.get(agent_id)
    if not agent:
        logger.error(f"Agent not found: {agent_id}")
        mcp.update_task_status(task_id, "failed", {"error": f"Agent not found: {agent_id}"})
        return
    
    # Prepare payload for agent
    payload = {
        "task_id": task_id,
        "action": task.get("type", "process"),
        "parameters": task.get("parameters", {}),
        "session_id": task.get("session_id")
    }
    
    # Call agent
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent["endpoint_url"], 
                json=payload, 
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                mcp.update_task_status(task_id, "completed", result)
                logger.info(f"Task completed: {task_id}")
            else:
                error = f"Agent returned status {response.status_code}"
                mcp.update_task_status(task_id, "failed", {"error": error})
                logger.error(f"Task failed: {task_id} - {error}")
                
    except Exception as e:
        error = f"Error processing task: {str(e)}"
        mcp.update_task_status(task_id, "failed", {"error": error})
        logger.error(f"Task exception: {task_id} - {error}")

# Run server
if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8000"))
    logger.info(f"Starting MCP server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)