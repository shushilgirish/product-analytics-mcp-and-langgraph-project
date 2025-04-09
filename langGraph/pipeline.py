"""
Criminal Report Generation Pipeline using LangGraph
Integrates multiple agents for comprehensive criminal report analysis
"""

import os
import sys
import operator
import traceback
import json
from typing import TypedDict, Dict, Any, List, Annotated, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import tempfile
import requests
import re
import os
from openai import OpenAI
# Add to imports at top of file
import numpy as np
import scipy.stats as stats
# LangChain and LangGraph imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from graphviz import Digraph

# Add parent directory to path for imports from other project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules
from agents.websearch_agent import tavily_search, tavily_extract, build_markdown_report
from agents.rag_agent import RAGAgent
# Make sure this is at the top
from agents.snowflake_utils import CrimeDataAnalyzer, initialize_connections, CrimeReportRequest
from agents.Comparision_agent import ComparisonAgent
from agents.forecast_code_agent import ForecastAgent
from agents.judge_agent import JudgeAgent
from agents.llmselection import LLMSelector as llmselection
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

###############################################################################
# State definition for Criminal Report Generation
###############################################################################
class CrimeReportState(TypedDict, total=False):
    """State definition for the crime report generation pipeline"""
    # Input parameters
    question: str  # User's query (removing redundant 'input')
    search_mode: str  # "all_years" or "specific_range"
    start_year: Optional[int] 
    end_year: Optional[int] 
    selected_regions: List[str]  # Cities/regions to analyze
    model_type: str  # LLM model to use
    
    # Agent outputs
    web_output: Dict[str, Any]  # Results from WebSearchAgent
    rag_output: Dict[str, Any]  # Results from RAGAgent
    snowflake_output: Dict[str, Any]  # Results from CrimeDataAnalyzer
    comparison_output: Dict[str, Any]  # Results from ComparisonAgent
    forecast_output: Dict[str, Any]  # Crime trend forecasts
    #Report organization 
    report_sections: Dict[str, Any]  # Structured report sections with content
    visualizations: Dict[str, Any]  # All visualizations from different sources
    contextual_images: Dict[str, Any]  # Contextual images generated for the report
    safety_assessment: Dict[str, Any]  # Safety assessment results
    
    # Processing metadata
    chat_history: List[Dict[str, Any]]  # Conversation history with feedback
    intermediate_steps: Annotated[List[tuple[Any, str]], operator.add]  # Agent reasoning steps
    
    # Final outputs and evaluation
    final_report: Dict[str, Any]  # Final structured report including:
                                 # - sections
                                 # - visualizations
                                 # - metadata
                                 # - quality assessment
    token_usage: List[Dict[str, Any]]  # Token usage tracking for each node
    
###############################################################################
# Node Functions
###############################################################################
def start_node(state: CrimeReportState) -> Dict:
    """Initial node that processes the input query and initializes the pipeline."""
    print(f"\n🚀 Starting crime report generation for: {state['question']}")
    print(f"📊 Analysis parameters:")
    print(f"- Regions: {', '.join(state['selected_regions'])}")
    print(f"- Time period: {state['search_mode']}")
    if state['search_mode'] == "specific_range":
        print(f"- Years: {state['start_year']} - {state['end_year']}")
    print(f"- Model: {state['model_type']}")
    
    return {
        "question": state["question"],
        "search_mode": state["search_mode"],
        "selected_regions": state["selected_regions"],
        "model_type": state["model_type"],
        "start_year": state.get("start_year"),
        "end_year": state.get("end_year")
    }


def web_search_node(state: CrimeReportState) -> Dict:
    """Execute web search for crime data using websearch_agent functions."""
    try:
        print("\n🔍 Executing web search for latest crime reports and news...")
        
        # Create query based on state parameters
        query = state["question"]
        if state["search_mode"] == "specific_range":
            query += f" between {state['start_year']} and {state['end_year']}"
        
        # Execute tavily search using functions from websearch_agent.py
        search_response = tavily_search(
            query=query,
            selected_regions=state["selected_regions"],
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            search_mode=state["search_mode"],
            topic="news",
            max_results=8
        )
        
        if not search_response:
            raise Exception("Search failed - no response from Tavily API")
            
        # Extract content from URLs
        urls = [item["url"] for item in search_response.get("results", []) 
                if "url" in item]
        extract_response = tavily_extract(urls=urls)
        
        # Build the report
        result_json = build_markdown_report(query, search_response, extract_response)
        result = json.loads(result_json)
        
        print(f"✅ Web search complete - found {result['metadata']['result_count']} results")
        return {"web_output": result}
    
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        traceback.print_exc()
        return {"web_output": {
            "markdown_report": f"Error during web search: {str(e)}",
            "images": [],
            "links": [],
            "metadata": {"error": str(e)}
        }}

def rag_node(state: CrimeReportState) -> Dict:
    """Execute RAG analysis for historical crime data."""
    try:
        print("\n📚 Retrieving historical crime data using RAG...")
        
        # Get the model type from state
        model_type = state.get("model_type")
        if model_type:
            print(f"RAG node using model: {model_type}")
        
        # Initialize the RAG agent with the user-selected model
        rag_agent = RAGAgent(model_name=model_type)
        
        # Process the query with the RAG agent
        result = rag_agent.process(
            query=state["question"],
            search_mode=state.get("search_mode", "all_years"),
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            selected_regions=state.get("selected_regions", []) 
        )
        usage = track_token_usage(
            state=state,
            node_name="rag_analysis",
            input_text=state["question"],
            output_text=result.get("insights")
        )

        print(f"✅ RAG analysis complete using {result.get('model_used', model_type)} , used {usage['total_so_far']} tokens and cost {usage['total_cost_so_far']}$ ")
        return {
            "rag_output": result,
            }
        
    except Exception as e:
        print(f"❌ RAG analysis error: {str(e)}")
        traceback.print_exc()
        return {"rag_output": {"error": str(e), "status": "failed"}}
        
def snowflake_node(state: CrimeReportState) -> Dict:
    """Execute Snowflake analysis for crime data visualizations."""
    try:
        print("\n📊 Analyzing crime data from Snowflake...")
        
        # Validate regions (new validation)
        regions = state["selected_regions"]
        if not regions or regions == [""]:
            raise ValueError("No valid regions specified")
            
        # Create the request with validated regions
        request = CrimeReportRequest(
            question=state["question"],
            search_mode=state["search_mode"],
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            selected_regions=regions, 
            model_type=state["model_type"]
        )
        
        # Initialize connections
        engine, llm = initialize_connections(model_type=state["model_type"])
        analyzer = CrimeDataAnalyzer(engine, llm)
        
        # Execute analysis with error handling
        try:
            result = analyzer.analyze_crime_data(request)
            # Validate visualization paths
            if "visualizations" in result and "paths" in result["visualizations"]:
                paths = result["visualizations"]["paths"]
                validated_paths = {}
                for key, path in paths.items():
                    if os.path.exists(path):
                        validated_paths[key] = path
                    else:
                        print(f"⚠️ Missing visualization: {path}")
                result["visualizations"]["paths"] = validated_paths
            
            usage = track_token_usage(
                state=state,
                node_name="snowflake_analysis",
                input_text=state["question"],
                output_text=result.get("analysis")
            ) or {"tokens": 0}  # Provide default if tracking fails
            
            print(f"✅ Snowflake analysis complete with {len(result.get('visualizations', {}).get('paths', {}))} visualizations and used {usage['total_so_far']} tokens and {usage['total_cost_so_far']}$")
            return {"snowflake_output": result}
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {"snowflake_output": {
                "status": "failed",
                "error": str(e),
                "analysis": "Analysis failed due to an error"
            }}
            
    except Exception as e:
        print(f"❌ Snowflake node error: {str(e)}")
        return {"snowflake_output": {"status": "failed", "error": str(e)}}
        
def contextual_image_node(state: CrimeReportState) -> Dict:
    """Generate contextual images for the report."""
    try:
        print("\n🎨 Generating contextual images for the report...")
        
        # Load environment variables
        load_dotenv()
        XAI_API_KEY = os.getenv("GROK_API_KEY")
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
        
        # Define regions for context
        regions = state['selected_regions']
        regions_str = ", ".join(regions)
        
        # Define prompts for different contextual images
        prompts = [
            {
                "title": "Crime Prevention Strategies",
                "prompt": f"A photorealistic image showing modern crime prevention strategies in {regions_str}. Community watch programs, police presence, and advanced surveillance technology working together. Professional style for a data report.",
                "prefix": "crime_prevention"
            },
            {
                "title": f"Urban Safety in {regions[0]}",
                "prompt": f"A photorealistic image of urban safety features in {regions[0]}. Well-lit streets, security cameras, police patrols, and people feeling safe in public spaces. Professional style for a crime analysis report.",
                "prefix": "urban_safety"
            },
            {
                "title": "Community Policing Impact",
                "prompt": "A photorealistic image of effective community policing. Police officers interacting positively with diverse community members, participating in neighborhood events, and building trust. Professional style for a crime analysis report.",
                "prefix": "community_policing"
            }
        ]
        
        # Generate images for each prompt
        contextual_images = {}
        
        for prompt_data in prompts:
            print(f"\n🎨 Generating image for: {prompt_data['prefix']}")
            print(f"Prompt: {prompt_data['prompt']}")
            
            try:
                # Request image from X.AI
                response = client.images.generate(
                    model="grok-2-image-1212",
                    prompt=prompt_data['prompt'],
                    n=1
                )
                
                # Get the image URL
                image_url = response.data[0].url
                print(f"✅ Image URL: {image_url}")
                
                # Download the image
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    # Create timestamped filename
                    image_path = f"{prompt_data['prefix']}.png"
                    
                    # Save the image
                    img = Image.open(BytesIO(img_response.content))
                    img.save(image_path)
                    print(f"✅ Image saved to: {image_path}")
                    
                    contextual_images[prompt_data['title']] = {
                        "path": image_path,
                        "prompt": prompt_data['prompt'],
                        "description": "AI-generated image",
                        "rationale": f"Illustrative image for {prompt_data['title']}"
                    }
            except Exception as e:
                print(f"❌ Error generating image for {prompt_data['title']}: {str(e)}")
        # Track token usage for prompt generation
        prompt_text = "\n".join([p["prompt"] for p in prompts])
        usage = track_token_usage(
            state=state,
            node_name="contextual_image_generation",
            input_text=prompt_text,
            output_text="Image generation output"  # Images don't have output tokens in the same way
        )
        tokens_used = usage.get("total_so_far", 0)
        total_cost = usage.get("total_cost_so_far", 0)
        # Append image-specific metadata to the usage tracking
        image_usage = {
            "image_count": len(contextual_images),
            "image_generation_model": "grok-2-image-1212",
            
            "image_generation_cost": len(contextual_images) * 0.25  # Example cost of $0.25 per image
        }

        # Update the token usage entry with image-specific information
        if "token_usage" in state and isinstance(state["token_usage"], list):
            for usage_entry in state["token_usage"]:
                if usage_entry.get("node") == "contextual_image_generation":
                    usage_entry.update(image_usage)
                    break
        # Print summary
        print(f"\n✅ Generated {len(contextual_images)} contextual images with {tokens_used} tokens used and {total_cost}$ cost")
        for title, data in contextual_images.items():
            print(f"- {title}: {data['path']}")
            
        return {"contextual_images": contextual_images}
        
    except Exception as e:
        print(f"❌ Contextual image generation error: {str(e)}")
        traceback.print_exc()
        return {"contextual_images": {}}

def comparison_node(state: CrimeReportState) -> Dict:
    """Create comparative analysis using ComparisonAgent with memory."""
    try:
        print("\n🔄 Creating comparative analysis...")
        
        # Get or create comparison agent
        if not hasattr(comparison_node, 'agent'):
            comparison_node.agent = ComparisonAgent(state["model_type"])
        
        # Prepare analysis request
        analysis_request = {
            "regions": state["selected_regions"],
            "snowflake_data": state.get("snowflake_output", {}),
            "rag_data": state.get("rag_output", {}),
            "web_data": state.get("web_output", {})
        }
        
        # Execute analysis
        comparison_type = "cross_region" if len(state["selected_regions"]) > 1 else "temporal"
        result = comparison_node.agent.analyze(analysis_request, comparison_type)
        
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            if viz_data := state["snowflake_output"].get("visualizations", {}):
                result["visualizations"] = {
                    **{f'top5_incidents_{city.replace(" ", "_")}': viz_data.get("paths", {}).get(f'top5_incidents_{city.replace(" ", "_")}') 
                       for city in state["selected_regions"]},
                    "yearly_distribution": viz_data.get("paths", {}).get("yearly_distribution")
                }
        
        usage  = track_token_usage(
            state=state,
            node_name="comparison_analysis",
            input_text=state["question"],
            output_text=result.get("comparison")
        )
        viz_count = len(result.get("visualizations", {}))
        print(f"✅ Comparison analysis complete with {viz_count} visualizations,  {usage['total_so_far']} total tokens and {usage['total_cost_so_far']}$ cost")
        return {"comparison_output": result}
        
    except Exception as e:
        print(f"❌ Comparison analysis error: {str(e)}")
        traceback.print_exc()
        return {"comparison_output": {
            "error": str(e),
            "status": "failed"
        }}
    
def forecast_node(state: CrimeReportState) -> Dict:
    """Generate crime trend forecasts for future periods and forecasting code."""
    try:
        print("\n🔮 Generating crime trend forecasts...")
        
        
        if "snowflake_output" not in state or state["snowflake_output"].get("status") != "success":
            print("⚠️ Skipping forecast - no valid data available")
            return {"forecast_output": {
                "status": "skipped",
                "reason": "No valid data available for forecasting"
            }}
        
        
        llm = llmselection.get_llm(state["model_type"])
        
        
        stats = state["snowflake_output"]["statistics"]
        yearly_trends = stats["incident_analysis"]["yearly_trends"]
        
        # Get insights from RAG and web search
        rag_insights = state.get("rag_output", {}).get("insights", "No historical insights available")
        web_trends = state.get("web_output", {}).get("markdown_report", "No web search data available")
        
        # Limit text size to avoid token issues
        rag_insights_sample = rag_insights[:1000] + "..." if len(rag_insights) > 1000 else rag_insights
        web_trends_sample = web_trends 
        
        # Convert to a format the LLM can understand
        trend_data = json.dumps(yearly_trends, indent=2)
        
        # Create prompt for forecasting with additional context
        forecast_prompt = f"""
        You are a crime data forecasting expert. Based on the historical crime data and insights provided,
        generate future crime trend forecasts for {', '.join(state['selected_regions'])}.
        
        Historical Crime Data:
        {trend_data}
        
        Historical Context which shows the summary of the crime that happened(from RAG):
        {rag_insights_sample}
        
        Recent News and Trends (from Web):
        {web_trends_sample}
        
        Please provide:
        1. Short-term forecast (next 1-2 years)
        2. Medium-term forecast (3-5 years)
        3. Key indicators to monitor
        4. Potential intervention points
        
        Format your response with clear headings and bullet points.
        """
        
        # Generate forecast
        forecast = llmselection.get_response(llm, forecast_prompt)
        
        
        code_generator = ForecastAgent(model_type=state["model_type"])
        forecast_data = code_generator.generate_forecast_data(
            snowflake_data=state["snowflake_output"],
            rag_data=state.get("rag_output", {}),
            web_data=state.get("web_output", {}),
            comparison_data=state.get("comparison_output", {}),
            regions=state["selected_regions"]
        )
        
        
        forecast_output = {
            "forecast": forecast,  
            "forecast_data": {
                "historical": forecast_data.get("historical", {}),
                "forecast": forecast_data.get("forecast", {}),
                "combined": forecast_data.get("combined", {}),
                "metadata": forecast_data.get("metadata", {})
            },
            "status": "success"
        }
        
        usage = track_token_usage(
            state=state,
            node_name="forecast_generation",
            input_text=forecast_prompt,
            output_text=forecast
        )
        tokens_used = usage.get("total_so_far")
        print(f"Total tokens so far: {tokens_used} and cost {usage['total_cost_so_far']}$")
        return {"forecast_output": forecast_output}
        
    except Exception as e:
        print(f"❌ Forecast generation error: {str(e)}")
        traceback.print_exc()
        return {"forecast_output": {"error": str(e), "status": "failed"}}


def safety_assessment_node(state: CrimeReportState) -> Dict:
    """Generate safety assessment and recommendations."""
    try:
        print("\n🛡️ Generating safety assessment...")
        
        # Use the LLM to generate safety assessment
        llm = llmselection.get_llm(state["model_type"])
        
        # Create a simple safety assessment agent
        safety_tool = Tool(
            name="safety_assessment",
            description="Assess safety based on crime data",
            func=lambda x: x  
        )
        
        safety_agent = initialize_agent(
            tools=[safety_tool],
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=1
        )
        
        # Gather data from previous nodes
        web_data = state.get("web_output", {}).get("markdown_report", "No web data available")
        rag_data = state.get("rag_output", {}).get("insights", "No historical insights available")
        snowflake_data = state.get("snowflake_output", {}).get("analysis", "No analysis available")
        
        # Create prompt for safety assessment
        safety_prompt = f"""
        You are a public safety expert. Based on the crime data and analysis provided,
        generate a comprehensive safety assessment for {', '.join(state['selected_regions'])}.
        
        Web Search Data:
        {web_data[:1000]}...
        
        Historical Insights:
        {rag_data[:700]}
        
        Statistical Analysis:
        {snowflake_data[:1000]}
        
        Please provide:
        1. Current Safety Rating (scale 1-10)
        2. High-Risk Areas and Times
        3. Vulnerable Demographics
        4. Safety Recommendations for Residents
        5. Recommendations for Law Enforcement
        6. Potential Policy Interventions
        
        Format your response with clear headings and bullet points.
        """
        
        # Generate safety assessment using the agent
        safety_assessment = safety_agent.run(safety_prompt)
        usage = track_token_usage(
            state=state,
            node_name="safety_assessment",
            input_text=safety_prompt,
            output_text=safety_assessment
        )
        tokens_used = usage.get("tokens", 0)
        total_cost = usage.get("total_cost_so_far", 0)
        
        print(f"✅ Safety assessment complete with {tokens_used} tokens) and {total_cost}$ cost")
        return {"safety_assessment": safety_assessment}
        
    except Exception as e:
        print(f"❌ Safety assessment error: {str(e)}")
        traceback.print_exc()
        return {"safety_assessment": f"Error generating safety assessment: {str(e)}"}


def report_organization_node(state: CrimeReportState) -> Dict:
    """Organize all data into structured report sections and generate missing content."""
    try:
        print("\n📝 Organizing report sections and synthesizing content...")
        
        # Define the report structure with empty sections
        report_sections = {
            "executive_summary": {
                "title": "Executive Summary",
                "content": "",
                "order": 1,
                "images": []
            },
            "methodology": {
                "title": "Methodology and Data Sources",
                "content": "",
                "order": 2,
                "images": []
            },
            "historical_context": {
                "title": "Historical Context and Trends",
                "content": "",
                "order": 3,
                "images": []
            },
            "current_analysis": {
                "title": "Current Crime Landscape Analysis",
                "content": "",
                "order": 4,
                "visualizations": [],
                "images": []
            },
            "regional_comparison": {
                "title": "Regional Comparison Analysis",
                "content": "",
                "order": 5,
                "visualizations": [],
                "images": []
            },
            "safety_assessment": {
                "title": "Safety Assessment",
                "content": state.get("safety_assessment", ""),
                "order": 6,
                "images": []
            },
            "forecast": {
                "title": "Crime Trend Forecast",
                "content": state.get("forecast_output", {}).get("forecast"),
                "dfs": [],
                "order": 7,
                "images": []
            },
            "recommendations": {
                "title": "Recommendations and Interventions",
                "content": "",
                "order": 8,
                "images": []
            },
            "appendix": {
                "title": "Appendix: Additional Data and Visualizations",
                "content": "",
                "order": 9,
                "visualizations": [],
                "images": [],
                "links": []
            }
        }
        
        
        
        # Historical context from RAG output
        if "rag_output" in state:
            historical_insights = state["rag_output"].get("insights", "")
            report_sections["historical_context"]["content"] = historical_insights
            
    
        
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            # Current analysis
            viz_paths = state["snowflake_output"].get("visualizations", {}).get("paths", {})
            if all_incidents_path := viz_paths.get("all_incidents_trend"):
                report_sections["current_analysis"]["visualizations"].append(all_incidents_path)
            
            # Regional comparison
            comp_paths = state["snowflake_output"].get("visualizations", {}).get("comparison_paths", {})
            # Add them to the "regional_comparison" section
            for _, path in comp_paths.items():
                report_sections["regional_comparison"]["visualizations"].append(path)

        
        # Safety assessment from safety output
        if "safety_assessment" in state:
            report_sections["safety_assessment"]["content"] = state["safety_assessment"]
        
        # Forecast from forecast output
        if "forecast_output" in state and state["forecast_output"].get("status") == "success":
            forecast = state["forecast_output"].get("forecast", "")
            report_sections["forecast"]["content"] = forecast
            for key, value in state["forecast_output"].get("forecast_data", {}).items():
                if key == "historical":
                    report_sections["forecast"]["dfs"].append(value)
                elif key == "forecast":
                    report_sections["forecast"]["dfs"].append(value)
                elif key == "combined":
                    report_sections["forecast"]["dfs"].append(value)

            
        
        # Methodology - standard content based on what was used
        methodology_content = [
            f"This report analyzes crime data for {', '.join(state['selected_regions'])} using multiple data sources:",
            "- Historical crime records through Retrieval Augmented Generation",
            "- Statistical analysis using Snowflake database",
            "- Latest news articles and reports from web searches",
            "- Comparative analysis across regions and time periods",
            "- AI-driven forecasting and trend analysis",
            "",
            f"The analysis covers {state.get('start_year', 'all available history')} to {state.get('end_year', 'present')}."
        ]
        report_sections["methodology"]["content"] = "\n".join(methodology_content)
        
    
        llm = llmselection.get_llm(state["model_type"])
        
        # Generate missing content for empty sections
        for section_key, section in report_sections.items():
            if not section.get("content"):
                print(f"Generating content for {section.get('title')}")
                
                if section_key == "executive_summary":
                    # Create an executive summary based on all available data
                    summary_prompt = f"""
                    Create a concise executive summary (max 250 words) of the crime analysis report for {', '.join(state['selected_regions'])}. 
                    Include key findings about crime rates, patterns, and notable trends.
                    Focus on the most important insights that a decision-maker would need to know.
                    """
                    executive_summary_content = llmselection.get_response(llm, summary_prompt)
                    section["content"] = executive_summary_content
                    token_usage = track_token_usage(
                        state=state,
                        node_name="executive_summary",
                        input_text=summary_prompt,  
                        output_text=executive_summary_content
                    )
                    print(f"✅ Executive summary generated with {token_usage['total_so_far']} tokens and {token_usage['total_cost_so_far']}$ cost")

                elif section_key == "recommendations":
                    # Generate recommendations based on all analyses
                    safety_assessment = state.get("safety_assessment", "")
                    forecast = state.get("forecast_output", {}).get("forecast", "")
                    
                    recommendations_prompt = f"""
                    Based on the crime data and analysis for {', '.join(state['selected_regions'])}, 
                    provide specific, actionable recommendations for:
                    1. Law enforcement strategies
                    2. Community safety measures
                    3. Policy interventions
                    
                    Safety Assessment: {safety_assessment[:500]}...
                    
                    Forecast Insights: {forecast[:500]}...
                    
                    Format with clear bullet points and prioritize by potential impact.
                    """
                    section["content"] = llmselection.get_response(llm, recommendations_prompt)

                    token_usage = track_token_usage(
                        state=state,
                        node_name="recommendations",
                        input_text=recommendations_prompt,  
                        output_text=section["content"]
                    )
                    print(f"✅ Recommendations generated with {token_usage['total_so_far']} tokens and {token_usage['total_cost_so_far']}$ cost")
                
                elif section_key == "appendix":
                    # Generate appendix content
                    appendix_prompt = f"""
                    Create a brief appendix section for a crime report including:
                    1. Data sources and methodologies
                    2. Statistical methods used
                    3. Glossary of crime-related terms
                    4. References
                    """
                    section["content"] = llmselection.get_response(llm, appendix_prompt)
                    token_usage = track_token_usage(
                        state=state,
                        node_name="appendix",
                        input_text=appendix_prompt,  
                        output_text=section["content"]
                    )
                    print(f"✅ Appendix content generated with {token_usage['total_so_far']} tokens and {token_usage['total_cost_so_far']}$ cost")
        visualizations = {}
        
        # Add Snowflake visualizations
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            viz_paths = state["snowflake_output"].get("visualizations", {}).get("paths", {})
            for viz_type, path in viz_paths.items():
                visualizations[f"snowflake_{viz_type}"] = path
    
        
        # Add contextual images to appropriate sections
        if "contextual_images" in state and state["contextual_images"]:
            # Match contextual images to sections by keywords
            for title, image_data in state["contextual_images"].items():
                if "prevention" in title.lower():
                    report_sections["recommendations"]["images"].append(image_data)
                elif "safety" in title.lower():
                    report_sections["safety_assessment"]["images"].append(image_data)
                elif "policing" in title.lower():
                    report_sections["current_analysis"]["images"].append(image_data)
                else:
                    # Default to executive summary
                    report_sections["executive_summary"]["images"].append(image_data)
        
        # Add web search images to appendix
        if "web_output" in state:
            # Add images
            images = state["web_output"].get("images", [])
            for i, img_url in enumerate(images):
                visualizations[f"web_image_{i}"] = img_url
                report_sections["appendix"]["visualizations"].append(img_url)
            
            # Add links to the appendix
            links = state["web_output"].get("links", [])
            if links and isinstance(links, list):
                # Format links for appendix
                link_content = ["### Reference Links", ""]
                for i, link_data in enumerate(links, start=1):
                    if isinstance(link_data, dict):
                        title = link_data.get("title", "Untitled")
                        url = link_data.get("url", "#")
                        source = link_data.get("source", "Unknown")
                        pub_date = link_data.get("published_date", "Unknown")
                        link_content.append(f"{i}. [{title}]({url}) - {source} ({pub_date})")
                    elif isinstance(link_data, str):
                        link_content.append(f"{i}. [{link_data}]({link_data})")
                
                # Store the formatted links in the appendix section
                report_sections["appendix"]["links"] = links
                
                # Add link details to appendix content
                if report_sections["appendix"]["content"]:
                    report_sections["appendix"]["content"] += "\n\n" + "\n".join(link_content)
                else:
                    report_sections["appendix"]["content"] = "\n".join(link_content)
        
        print(f"✅ Report organization and content synthesis complete with {len(visualizations)} visualizations")
        return {
            "report_sections": report_sections,
            "visualizations": visualizations,
            "synthesis_complete": True  # Keep this flag for pipeline flow control
        }
        
    except Exception as e:
        print(f"❌ Report organization error: {str(e)}")
        traceback.print_exc()
        return {"report_sections": {}, "visualizations": {}}
        

def final_report_node(state: CrimeReportState) -> Dict:
    """Assemble the final report with all sections."""
    try:
        print("\n📊 Generating final crime report...")
        
        # Get report sections and visualizations
        report_sections = state.get("report_sections", {})
        visualizations = state.get("visualizations", {})
        contextual_images = state.get("contextual_images", {})

        token_usage = state.get("token_usage", {
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_node": {}
        })
        
        # Log once
        section_titles = [section.get("title") for section in report_sections.values()]
        print(f"Found {len(section_titles)} sections")
        print(f"Found {len(visualizations)} visualizations")
        print(f"Found {len(contextual_images)} contextual images")
        
        # Create final report structure
        final_report = {
            "title": f"Crime Analysis Report: {', '.join(state['selected_regions'])}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections": sorted([section for section in report_sections.values()], 
                             key=lambda x: x.get("order", 999)),
            "visualizations": list(visualizations.values()) if visualizations else [],
            "contextual_images": list(contextual_images.values()) if contextual_images else [],
            "token_usage_summary": {  # Add token usage summary
                "total_tokens": token_usage.get("total_tokens", 0),
                "total_cost": token_usage.get("total_cost", 0),
                "by_node": token_usage.get("by_node", {}),
                "model_info": token_usage.get("model_used", {})
            }
        }
        
        # Generate cover
        if cover_image_path := generate_report_cover(
            title=final_report["title"],
            regions=state["selected_regions"],
            time_period=f"{state['start_year']} to {state['end_year']}" if state.get("search_mode") == "specific_range" else "all years"
        ):
            final_report["cover_image"] = cover_image_path
            print(f"✅ Cover image generated: {cover_image_path}")
        
        
        
        
        
        return {"final_report": final_report}
        
    except Exception as e:
        print(f"❌ Final report generation error: {str(e)}")
        traceback.print_exc()
        return {"final_report": {
            "error": str(e),
            "title": "ERROR: Report Generation Failed",
            "sections": [{"title": "Error Details", "content": str(e)}]
        }}
            
def judge_node(state: CrimeReportState) -> Dict:
    """Evaluate report quality using JudgeAgent and merge the evaluation and token usage into the final report."""
    try:
        print("\n⚖️ Evaluating report quality...")
        # Initialize the Judge Agent if not already done
        if not hasattr(judge_node, 'agent'):
            judge_node.agent = JudgeAgent(model_type=state["model_type"])
            print(f"Judge agent initialized with model: {state['model_type']}")
        
        # Ensure that a final report exists in state
        if "final_report" not in state:
            raise ValueError("Final report not found in state")
        
        # Create an evaluation context from the final report and input parameters
        evaluation_context = {
            'report': state["final_report"],
            'regions': state["selected_regions"],
            'time_period': f"{state.get('start_year', 'all history')} to {state.get('end_year', 'present')}"
        }
        
        # Evaluate the report using the Judge Agent
        evaluation = judge_node.agent.evaluate(evaluation_context)
        if not evaluation:
            raise ValueError("Evaluation returned no results")
        
        # Track token usage for this evaluation step
        usage = track_token_usage(
            state=state,
            node_name="report_evaluation",
            input_text=str(evaluation_context),
            output_text=str(evaluation)
        )
        print(f"✅ Report evaluation complete with {usage.get('total_tokens', 0)} tokens and cost ${usage.get('cost', 0):.4f}")
        
        # Merge the evaluation output into the final report
        state["final_report"]["evaluation"] = evaluation
        
        if "token_usage" in state:
            state["final_report"]["token_usage_summary"] = {
                "total_tokens": state["token_usage"].get("total_tokens", 0),
                "total_cost": state["token_usage"].get("total_cost", 0.0),
                "by_node": state["token_usage"].get("by_node", {}),
                "model_info": state["token_usage"].get("model_used", {})
            }
        else:
            state["final_report"]["token_usage_summary"] = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_node": {},
                "model_info": {}
            }
        
        return evaluation
    except Exception as e:
        print(f"❌ Report evaluation error: {str(e)}")
        return {"judge_feedback": {"error": str(e)}}
        
    
###############################################################################
# Helper Functions for Visualization
###############################################################################
def track_token_usage(state: CrimeReportState, node_name: str, input_text: str, output_text: str) -> Dict:
    """Track token usage for each node and maintain running totals with pricing."""
    try:
        model_name = state["model_type"]
        
        
        input_tokens = llmselection.count_tokens(input_text, model_name)
        output_tokens = llmselection.count_tokens(output_text, model_name)
        total_tokens = input_tokens + output_tokens
        
        
        model_info = llmselection.get_token_limits(model_name)
        cost_per_1k = model_info["cost_per_1k"]
        context_window = model_info["context_window"]
        
        
        cost = (total_tokens / 1000) * cost_per_1k
        
        # Initialize token tracking if not present
        if "token_usage" not in state:
            state["token_usage"] = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_node": {},
                "model_used": {
                    "name": model_name,
                    "context_window": context_window,
                    "cost_per_1k": cost_per_1k
                }
            }
        
        # Initialize node if not present
        if node_name not in state["token_usage"]["by_node"]:
            state["token_usage"]["by_node"][node_name] = {
                "tokens": 0,
                "cost": 0.0
            }
        
        # Update node usage
        state["token_usage"]["by_node"][node_name]["tokens"] += total_tokens
        state["token_usage"]["by_node"][node_name]["cost"] += cost
        
        # Update total usage
        state["token_usage"]["total_tokens"] += total_tokens
        state["token_usage"]["total_cost"] += cost
        
        return {
            "node": node_name,
            "tokens": total_tokens,
            "cost": cost,
            "total_so_far": state["token_usage"]["total_tokens"],
            "total_cost_so_far": state["token_usage"]["total_cost"],
            "model_info": {
                "name": model_name,
                "context_window": context_window,
                "cost_per_1k": cost_per_1k
            }
        }
        
    except Exception as e:
        print(f"Error tracking tokens: {str(e)}")
        return {
            "node": node_name,
            "tokens": 0,
            "cost": 0.0,
            "total_so_far": 0,
            "total_cost_so_far": 0.0,
            "model_info": None,
            "error": str(e)
        }




def embed_image_as_base64(image_path: str) -> Optional[str]:
    """Convert image to base64 string with proper error handling and format detection."""
    try:
        if not os.path.exists(image_path):
            print(f"⚠️ Image file not found: {image_path}")
            return None

        with open(image_path, "rb") as image_file:
            # Read image content
            image_content = image_file.read()
            
            # Detect image format
            img = Image.open(BytesIO(image_content))
            img_format = img.format.lower() if img.format else 'png'
            
            # Convert to base64
            encoded_string = base64.b64encode(image_content).decode("utf-8")
            return f"data:image/{img_format};base64,{encoded_string}"
            
    except Exception as e:
        print(f"❌ Error embedding image {image_path}: {str(e)}")
        return None

# Then modify the markdown generation part in final_report_node:
def generate_report_cover(title: str, regions: List[str], time_period: str) -> str:
    """Generate a cover image for the report with title and key info."""
    try:
        # Create a blank image
        width, height = 800, 1100
        cover = Image.new('RGB', (width, height), color=(245, 245, 245))
        draw = ImageDraw.Draw(cover)
        
        try:
            # Try to load a nice font, fall back to default if not available
            title_font = ImageFont.truetype("Arial.ttf", 40)
            subtitle_font = ImageFont.truetype("Arial.ttf", 30)
            info_font = ImageFont.truetype("Arial.ttf", 20)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # Add a decorative header bar
        draw.rectangle([(0, 0), (width, 100)], fill=(30, 50, 100))
        
        # Add title
        title_wrapped = "\n".join([title[i:i+30] for i in range(0, len(title), 30)])
        draw.text((width/2, 200), title_wrapped, font=title_font, fill=(30, 50, 100), anchor="mm")
        
        # Add regions
        regions_text = "Regions Analyzed: " + ", ".join(regions)
        draw.text((width/2, 300), regions_text, font=subtitle_font, fill=(60, 80, 120), anchor="mm")
        
        # Add time period
        period_text = f"Time Period: {time_period}"
        draw.text((width/2, 350), period_text, font=subtitle_font, fill=(60, 80, 120), anchor="mm")
        
        # Add generation info
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((width/2, 450), f"Generated: {gen_time}", font=info_font, fill=(100, 100, 100), anchor="mm")
        
        # Add system info
        draw.text((width/2, 480), "Multi-Agent Crime Analysis System", font=info_font, fill=(100, 100, 100), anchor="mm")
        
        # Add a decorative footer
        draw.rectangle([(0, height-50), (width, height)], fill=(30, 50, 100))
        
        # Save the cover
        output_path = f"report_cover.png"
        cover.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error generating report cover: {e}")
        return ""
    

###############################################################################
# Pipeline Building
###############################################################################
def build_pipeline():
    """Build and compile the pipeline."""
    try:
        graph = StateGraph(CrimeReportState)
        def parallel_data_gathering(state: CrimeReportState) -> Dict:
            """Execute web search, RAG, and snowflake analysis in parallel."""
            print("\n🚀 Starting parallel data gathering...")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                web_future = executor.submit(web_search_node, state)
                rag_future = executor.submit(rag_node, state)
                snowflake_future = executor.submit(snowflake_node, state)
                
                web_result = web_future.result()
                rag_result = rag_future.result()
                snowflake_result = snowflake_future.result()
            
            print("✅ Parallel data gathering complete")
            return {
                **web_result,
                **rag_result,
                **snowflake_result
            }
        # Add nodes
        graph.add_node("start", start_node)
        graph.add_node("parallel_gathering", parallel_data_gathering)
        graph.add_node("comparison", comparison_node)
        graph.add_node("forecast", forecast_node)
        graph.add_node("safety", safety_assessment_node)
        graph.add_node("contextual_img", contextual_image_node)
        graph.add_node("organization", report_organization_node)
        graph.add_node("report_generation", final_report_node)
        graph.add_node("judge", judge_node)
        
        # Set entry point
        graph.set_entry_point("start")

        # Define linear flow with completion checks
        def is_complete(state: Dict, key: str) -> bool:
            return key in state and state[key] is not None

        # Start -> Parallel Gathering (unconditional)
        graph.add_edge("start", "parallel_gathering")

        # Parallel Gathering -> Contextual Images
        graph.add_conditional_edges(
            "parallel_gathering",
            lambda x: is_complete(x, "snowflake_output"),
            {True: "contextual_img"}
        )

        # Rest of the pipeline in linear order
        graph.add_edge("contextual_img", "comparison")
        graph.add_edge("comparison", "forecast")
        graph.add_edge("forecast", "safety")
        graph.add_edge("safety", "organization")
        graph.add_edge("organization", "report_generation")
        graph.add_edge("report_generation", "judge")
        graph.add_edge("judge", END)

        return graph.compile()
        
    except Exception as e:
        print(f"❌ Error building pipeline: {str(e)}")
        return None
    

###############################################################################
# Main Invocation
###############################################################################
def cleanup_matplotlib():
    """Clean up matplotlib resources"""
    plt.close('all')

def generate_markdown_report(final_report: Dict, md_filename: str) -> None:
    try:
        # Validate input
        if not isinstance(final_report, dict):
            raise ValueError(f"Expected dict for final_report, got {type(final_report)}")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(md_filename) or '.', exist_ok=True)
        
        with open(md_filename, "w", encoding="utf-8") as f:
            # Write metadata with null checks
            f.write("\n".join([
                "---",
                f"title: {final_report.get('title', 'Crime Report')}",
                f"date: {final_report.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
                "---\n\n"
            ]))
            if cover_path := final_report.get('cover_image'):
                if os.path.exists(cover_path):
                    if cover_data := embed_image_as_base64(cover_path):
                        f.write(f"![Cover Image]({cover_data})\n\n")
                    else:
                        print("⚠️ Failed to embed cover image")

            # Write sections with validation
            sections = final_report.get("sections", [])
            if not sections:
                f.write("No sections found in report.\n\n")
            
            for section in sections:
                if not isinstance(section, dict):
                    continue
                    
                title = section.get("title", "Untitled Section")
                content = section.get("content", "No content available.")
                
                f.write(f"## {title}\n\n")
                f.write(f"{content}\n\n")
                
                # Handle visualizations with validation
                for viz in section.get("visualizations", []) or []:
                    if not viz:
                        continue
                    if isinstance(viz, str):
                        if viz.startswith(('http://', 'https://')):
                            f.write(f"![Visualization]({viz})\n\n")
                        elif os.path.exists(viz):
                            if img_data := embed_image_as_base64(viz):
                                f.write(f"![Visualization]({img_data})\n\n")
                
                # Handle images with validation
                for img in section.get("images", []) or []:
                    if not isinstance(img, dict):
                        continue
                    path = img.get("path")
                    if path and os.path.exists(path):
                        if img_data := embed_image_as_base64(path):
                            f.write(f"![{img.get('description', 'Image')}]({img_data})\n\n")

            # Write token usage with validation
            if usage := final_report.get("token_usage"):
                f.write(f"\n## Token Usage\n")
                f.write(f"- Total Tokens: {usage.get('total_tokens', 0):,}\n")
                f.write(f"- Total Cost: ${usage.get('total_cost', 0):.2f}\n\n")

    except Exception as e:
        print(f"❌ Error generating markdown: {str(e)}")

def main():
    try:
        if not (pipeline := build_pipeline()):
            raise Exception("Pipeline build failed")

        # Get and validate regions input
        regions_input = input("Regions (comma-separated) [default: Chicago, New York]: ").strip()
        regions = [r.strip() for r in regions_input.split(",")] if regions_input else ["Chicago", "New York"]
        regions = [r for r in regions if r]  
        
        if not regions:
            regions = ["Chicago", "New York"]
            print("⚠️ Using default regions: Chicago, New York")

        # Execute pipeline with validated regions
        result = pipeline.invoke({
            "question": "Analyze recent criminal incidents trends and patterns",
            "search_mode": "specific_range",
            "start_year": 2015,
            "end_year": 2024,
            "selected_regions": regions,
            "model_type": "Gemini Pro",
            "chat_history": [],
            "intermediate_steps": []
        })

        # Generate report
        md_filename = f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        generate_markdown_report(result.get("final_report"), md_filename)
        print(f"\n✅ Report generated: {md_filename}")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        return 1

    finally:
        cleanup_matplotlib()

if __name__ == "__main__":
    sys.exit(main())
