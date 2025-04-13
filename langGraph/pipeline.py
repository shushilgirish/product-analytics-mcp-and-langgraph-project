from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutorNode
from langchain_mcp.client import MCPClient

mcp_client = MCPClient(base_url="http://localhost:8001")
executor = ToolExecutorNode(mcp_client)

workflow = StateGraph()
workflow.add_node("tool_executor", executor)
