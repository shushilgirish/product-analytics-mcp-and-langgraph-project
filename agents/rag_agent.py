import logging
from langchain_core.tools import tool
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict, Any, Union, Optional, List
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import boto3
import traceback

# Initialize environment and configurations
load_dotenv(override=True)

SYSTEM_CONFIG = {
    "CURRENT_UTC": "2025-04-02 06:21:03",
    "CURRENT_USER": "user",
    "MIN_YEAR": 1995,
    "MAX_YEAR": 2018
}

# Initialize services
encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "crime-reports"))
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the LLMSelector from your llmselection file
from agents.llmselection import LLMSelector as llmselection

class SearchCrimeDataInput(BaseModel):
    tool_input: Dict[str, Any] = Field(..., description="The input parameters for the search")

class RAGAgent:
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the RAG agent with the specified model."""
        # Use passed model or default to Claude 3 Haiku
        self.model_name = model_name
        print(f"Initializing RAG Agent with model: {self.model_name}")
        
        self.llm = llmselection.get_llm(self.model_name)
        
        # Define the analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert crime analyst. Analyze the provided crime data to identify patterns, 
                         trends, and insights that would be valuable for law enforcement and public safety officials.
                         Your analysis should be comprehensive, focusing on:
                         - Key patterns in crime types and frequency
                         - Temporal trends (seasonal, yearly changes)
                         - Geographic distribution and hotspots
                         - Correlation with socioeconomic or other factors if available
                         - Potential causative factors
                         - Recommendations for intervention and prevention"""),
            ("user", """Please analyze the following crime data based on this query: {query}
                        
                        CRIME DATA:
                        {context}
                        
                        Provide a structured, insightful analysis with actionable recommendations.""")
        ])
        
        # Create the direct analysis chain
        self.analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
    
    def _analyze_data(self, query: str, context: str) -> str:
        """Internal method to analyze crime data using LLM."""
        try:
            if not context or len(context.strip()) < 100:
                return "Insufficient crime data to analyze."
                
            # Use the analysis chain to process the data
            return self.analysis_chain.invoke({
                "query": query, 
                "context": context[:5000]  # Limit context to avoid token limits
            })
        except Exception as e:
            logging.error(f"Error analyzing data: {str(e)}")
            traceback.print_exc()
            return f"Error analyzing crime data: {str(e)}"
    
    def process(self, query: str, search_mode: str = "all_years", 
                start_year: Optional[int] = None, end_year: Optional[int] = None, 
                selected_regions: Optional[List[str]] = None, 
                context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a crime data query using RAG techniques.
        
        Args:
            query: User's query about crime data
            search_mode: "all_years" or "specific_range"
            start_year: Start year for specific range
            end_year: End year for specific range
            selected_regions: List of regions to include
            context: Optional pre-retrieved context
            
        Returns:
            Dictionary with insights and metadata
        """
        try:
            # If context is already provided, use it directly
            if context:
                analysis = self._analyze_data(query=query, context=context)
                return {
                    "insights": analysis,
                    "analysis_type": "direct",
                    "model_used": self.model_name,
                    "status": "success"
                }
                
            # Otherwise, search for relevant data first
            search_input = {
                "query": query,
                "search_mode": search_mode,
                "start_year": start_year,
                "end_year": end_year,
                "model_type": self.model_name
            }
            
            # If regions are provided, include them
            if selected_regions:
                search_input["regions"] = selected_regions
                
            # Execute search to retrieve relevant crime data
            search_result = search_crime_data({"tool_input": search_input})
            
            # Handle string results (usually error messages)
            if isinstance(search_result, str):
                return {
                    "error": search_result,
                    "insights": "Failed to retrieve relevant crime data.",
                    "status": "failed"
                }
                
            # Extract the text context from search results
            formatted_context = search_result.get("raw_contexts", "No relevant crime data found.")
            
            # Analyze the context
            analysis = self._analyze_data(query=query, context=formatted_context)
            
            # Return comprehensive results
            return {
                "insights": analysis,
                "metadata": search_result.get("metadata", {}),
                "analysis_type": "search_and_analyze",
                "model_used": self.model_name,
                "status": "success",
                "raw_contexts": formatted_context[:1000] + "..." if len(formatted_context) > 1000 else formatted_context
            }
                
        except Exception as e:
            logging.error(f"RAG Agent processing error: {str(e)}")
            traceback.print_exc()
            return {
                "error": str(e),
                "insights": f"Error processing query: {str(e)}",
                "status": "failed"
            }

def get_chunk_from_s3(chunk_s3_path: str, chunk_index: int, s3_bucket: str) -> str:
    """Retrieve specific chunk data from S3 with improved error handling."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
            aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        response = s3_client.get_object(
            Bucket=s3_bucket,
            Key=chunk_s3_path
        )
        chunks_data = json.loads(response['Body'].read().decode('utf-8'))
        chunk_text = chunks_data.get(str(chunk_index), "")
        if len(chunk_text.strip()) < 100 or chunk_text.strip().startswith('#'):
            return ""
        return chunk_text
    except Exception as e:
        logging.error(f"Error retrieving chunk from S3: {e}")
        return ""

def rerank_results(query: str, results: list, top_k: int = 15) -> list:
    """Re-rank results using the cross-encoder for higher relevance."""
    if not results:
        return []
    passages = []
    for match in results:
        metadata = match['metadata']
        chunk_text = ""
        if metadata.get('chunks_s3_path') and metadata.get('chunk_index'):
            chunk_text = get_chunk_from_s3(
                chunk_s3_path=metadata['chunks_s3_path'],
                chunk_index=metadata['chunk_index'],
                s3_bucket=metadata.get('s3_bucket', 'crime-records')
            )
        text = chunk_text if chunk_text else metadata.get('text_preview', '').strip()
        if text:
            passages.append((match, text))
    if not passages:
        return []
    pairs = [[query, p[1]] for p in passages]
    scores = cross_encoder.predict(pairs)
    passage_scores = [(passages[i][0], score) for i, score in enumerate(scores)]
    reranked_results = sorted(passage_scores, key=lambda x: x[1], reverse=True)
    return [(item[0], float(item[1])) for item in reranked_results[:top_k]]

def format_results(matches: list) -> str:
    """Format search results with improved chunk data retrieval."""
    results = []
    for match, score in matches:
        metadata = match['metadata']
        chunk_text = ""
        if metadata.get('chunks_s3_path') and metadata.get('chunk_index'):
            chunk_text = get_chunk_from_s3(
                chunk_s3_path=metadata['chunks_s3_path'],
                chunk_index=metadata['chunk_index'],
                s3_bucket=metadata.get('s3_bucket', 'crime-records')
            )
        description = chunk_text if chunk_text else metadata.get('text_preview', '').strip()
        if len(description.strip()) < 50:
            continue
        results.append("\n".join([
            f"Year: {metadata.get('year', 'Unknown')}",
            f"Document ID: {metadata.get('document_id', 'Unknown')}",
            f"Description: {description}",
            f"Score: {score:.3f}\n"
        ]))
    return "\n---\n".join(results)

@tool("search_crime_data")
def search_crime_data(tool_input: Dict[str, Any]) -> Union[str, Dict]:
    """Search crime report data with improved retrieval and ranking."""
    try:
        # Extract query parameters from the tool input
        if isinstance(tool_input, dict) and "tool_input" in tool_input:
            input_params = tool_input["tool_input"]
        else:
            input_params = tool_input
            
        # Extract query parameters
        query = input_params.get("query")
        search_mode = input_params.get("search_mode", "all_years")
        start_year = input_params.get("start_year")
        end_year = input_params.get("end_year")
        model_type = input_params.get("model_type")
        regions = input_params.get("regions", [])
        
        if not query:
            return "Error: Query is required"
        
        # Encode query for vector search
        xq = encoder.encode([query])[0].tolist()
        
        # Determine year range based on search mode
        years_range = range(
            start_year or SYSTEM_CONFIG["MIN_YEAR"],
            (end_year or SYSTEM_CONFIG["MAX_YEAR"]) + 1
        ) if search_mode == "specific_range" else range(
            SYSTEM_CONFIG["MIN_YEAR"], 
            SYSTEM_CONFIG["MAX_YEAR"] + 1
        )
        
        # Collect initial results from each year namespace
        initial_results = []
        for year in years_range:
            try:
                response = index.query(
                    vector=xq,
                    top_k=10,
                    include_metadata=True,
                    namespace=str(year),
                    alpha=0.5
                )
                if response.get("matches"):
                    initial_results.extend(response["matches"])
            except Exception as e:
                logging.error(f"Error searching year {year}: {e}")
                continue
        
        if not initial_results:
            return "No results found for the specified time period."
        
        # Re-rank results using the cross-encoder
        reranked_results = rerank_results(query, initial_results, top_k=15)
        
        if not reranked_results:
            return "No quality results found after filtering."
            
        # Format results using improved chunk retrieval
        formatted_contexts = format_results(reranked_results)
        
        if not model_type:
            # Default to a reasonable model if none specified
            model_type = "Claude 3 Haiku"
            
        # Return the raw contexts and metadata
        return {
            "raw_contexts": formatted_contexts,
            "metadata": {
                "query": query,
                "search_mode": search_mode,
                "time_range": f"{min(years_range)}-{max(years_range)}",
                "timestamp": SYSTEM_CONFIG["CURRENT_UTC"],
                "user": SYSTEM_CONFIG["CURRENT_USER"],
                "result_count": len(reranked_results),
                "regions": regions
            }
        }
    except Exception as e:
        logging.error(f"Search failed: {e}")
        traceback.print_exc()
        return f"Error performing search: {str(e)}"

# Test the tool if run directly
if __name__ == "__main__":
    test_queries = [
        {
            "tool_input": {
                "query": "Spike in vehicle theft incidents",
                "search_mode": "specific_range",
                "start_year": 2000,
                "end_year": 2005
            }
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query['tool_input']['search_mode']}")
        try:
            # Get search results
            result = search_crime_data.invoke(query)
            if isinstance(result, dict):
                print(f"Success: Found {result.get('metadata', {}).get('result_count', 0)} results")
                
                # Create RAG agent and process the results using direct analysis
                rag_agent = RAGAgent("Claude 3 Haiku")
                processed_result = rag_agent.process(
                    query=query['tool_input']['query'],
                    context=result.get('raw_contexts'),
                    search_mode=query['tool_input']['search_mode'],
                    start_year=query['tool_input'].get('start_year'),
                    end_year=query['tool_input'].get('end_year')
                )
                
                # Save to file
                filename = f"crime_report_{query['tool_input']['search_mode']}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(processed_result, f, indent=2)
                print(f"Results saved to {filename}")
                
                # Now insights should be available
                print(f"Insights preview: {processed_result.get('insights', 'No insights')}...")
            else:
                print(f"Error: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()