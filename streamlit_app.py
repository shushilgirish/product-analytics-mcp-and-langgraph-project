import streamlit as st
import requests
import json
import toml
from streamlit_folium import st_folium
import folium
import os
import sys
import time
import base64
from datetime import datetime
# -------------------------------
# 1) Page Config for Wide Layout
# -------------------------------
st.set_page_config(
    page_title="Crime Analysis Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2) Configuration
# -------------------------------
config = toml.load("config.toml")
API_URL = config["fastapi_url"]
QUERY_URL = f"{API_URL}/generate_report"
MODELS_URL = f"{API_URL}/available_models"

# Add cities data after the API_URL definition
cities = {
    "New York": {"lat": 40.7128, "lon": -74.0060, "pop": 8419600, "crime_rate": 2.8},
    "San Francisco": {"lat": 37.7749, "lon": -122.4194, "pop": 883305, "crime_rate": 6.1},
    "Seattle": {"lat": 47.6062, "lon": -122.3321, "pop": 744955, "crime_rate": 5.2},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437, "pop": 3980400, "crime_rate": 3.9},
    "Houston": {"lat": 29.7604, "lon": -95.3698, "pop": 2328000, "crime_rate": 5.6},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "pop": 2716000, "crime_rate": 4.7},
}

# -------------------------------
# 3) Helper Functions
# -------------------------------
# Replace the problematic part with this code
def create_downloadable_report(report_data: dict) -> str:
    """Create a formatted markdown report for downloading"""
    try:
        # If report_data is already a string (markdown text)
        if isinstance(report_data, str):
            return report_data
            
        # Otherwise, format it as markdown from the dictionary
        timestamp = datetime.fromisoformat(report_data.get('timestamp', 
                     datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Crime Analysis Report

Generated: {timestamp}
Question: {report_data.get('question', 'N/A')}
Model: {report_data.get('model', 'N/A')}
Regions: {', '.join(report_data.get('selected_regions', []))}

## Quality Metrics
Overall Score: {report_data.get('judge_score', 0)}/10

### Detailed Metrics
{chr(10).join([f'- {metric}: {score}/10' for metric, score in report_data.get('judge_feedback', {}).items()])}

## Report Content
{report_data.get('content', '')}
"""
        return report
        
    except Exception as e:
        # Return a basic error report if something went wrong
        return f"# Error Creating Report\n\nThere was an error formatting the report: {str(e)}"
    
def handle_combined_report():
    st.title("Crime Analysis Assistant")
    st.subheader("💬 Research History")

    # Show chat history
    with st.container():
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                periods_text = "All Years" if message.get("search_type") == "all_years" else f"{message.get('start_year', '')} - {message.get('end_year', '')}"
                regions_text = ", ".join(message.get("selected_regions", []))
                
                # Create an expander for each message
                with st.expander(f"💬 Query #{i+1}: {message['content'][:50]}...", expanded=False):
                    st.markdown(f"""
                        <div class='user-message'>
                            <div class='metadata'>📅 {periods_text}<br>🌎 Regions: {regions_text}<br>🤖 Model: {message.get("model", "")}</div>
                            <div>🔍 {message['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # If there's a corresponding assistant message, show it in the same expander
                    if i + 1 < len(st.session_state.chat_history) and st.session_state.chat_history[i + 1]["role"] == "assistant":
                        assistant_message = st.session_state.chat_history[i + 1]
                        st.markdown(f"""
                            <div class='assistant-message'>
                                <div class='metadata'>🤖 Crime Analysis Assistant</div>
                                <div>{assistant_message['content']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add download button for each report
                        if "report_markdown" in assistant_message:
                            st.download_button(
                                label="📥 Download Report",
                                data=assistant_message["report_markdown"],
                                file_name=f"crime_report_{i}.md",
                                mime="text/markdown"
                            )

    # Check if we have selected cities
    if "selected_cities" not in st.session_state or not st.session_state["selected_cities"]:
        st.warning("⚠️ No cities selected. Please go to the Map View and select at least one city.")
        st.button("Go to Map View", on_click=nav_to, args=["Map View"])
        return

    # Input form
    st.markdown("---")
    with st.form("report_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input("Research Question", placeholder="What are the crime trends in New York?")
        with col2:
            submitted = st.form_submit_button("➤")

    if submitted and question:
        generate_report(question)

def generate_report(question):
    """Handle report generation and update chat history"""
    with st.spinner("🔄 Generating report..."):
        try:
            # Prepare request payload
            payload = {
                "question": question,
                "search_mode": "all_years" if st.session_state.search_type == "All Years" else "specific_range",
                "start_year": st.session_state.start_year,
                "end_year": st.session_state.end_year,
                "selected_regions": st.session_state["selected_cities"],
                "model_type": st.session_state.selected_model
            }

            # Add user message to chat history first
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
                "search_type": payload["search_mode"],
                "start_year": payload["start_year"],
                "end_year": payload["end_year"],
                "selected_regions": payload["selected_regions"],
                "model": payload["model_type"]
            })

            # Initiate report generation
            response = requests.post(QUERY_URL, json=payload)
            if response.status_code != 200:
                st.error(f"❌ API Error: {response.status_code}")
                return

            data = response.json()
            report_id = data["report_id"]
            status_url = data.get("status_url", f"/report_status/{report_id}")
            
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Poll for status
            for i in range(60):
                status_response = requests.get(f"{API_URL}{status_url}")
                if status_response.status_code != 200:
                    continue
                    
                status_data = status_response.json()
                if status_data["status"] == "completed":
                    progress_bar.progress(100)
                    status_text.write("✅ Report completed!")
                    break
                elif status_data["status"] == "failed":
                    st.error(f"Report generation failed: {status_data.get('error', 'Unknown error')}")
                    return
                else:
                    progress = min(5 + i * 1.5, 95)
                    progress_bar.progress(int(progress))
                    status_text.write(f"⏳ Processing report... ({int(progress)}%)")
                time.sleep(1)
            
            # Get the completed report
            report_response = requests.get(f"{API_URL}/report/{report_id}")
            if report_response.status_code != 200:
                st.error(f"❌ Failed to retrieve report: {report_response.status_code}")
                return
                
            content = report_response.text
            st.markdown(content, unsafe_allow_html=True)
            
            # Get evaluation and token usage data
            status_response = requests.get(f"{API_URL}/report_status/{report_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                evaluation = status_data.get("evaluation", {})
                token_summary = status_data.get("token_usage_summary", {})
                
                # Store report in session state
                report_data = {
                    "content": content,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "selected_regions": st.session_state["selected_cities"],
                    "model": st.session_state.selected_model,
                    "token_usage_summary": token_summary,
                    "final_report": status_data.get("final_report", {}),
                }
                st.session_state.reports[report_id] = report_data
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": content,
                    "report_id": report_id,
                    "evaluation": evaluation,
                    "token_usage": token_summary
                })
                
                # Display download and visualization options
                cols = st.columns(2)
                with cols[0]:
                    st.download_button(
                        label="📥 Download Report",
                        data=content,
                        file_name=f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                with cols[1]:
                    if st.button("Show Forecast Graph"):
                        final_report = st.session_state.reports[report_id].get("final_report", {})
                        st.write("Final report structure:", final_report.keys())  # Debug
                        forecast_section = final_report.get("forecast", {})
                        st.write("Forecast section structure:", forecast_section.keys())  # Debug
                        dfs_list = forecast_section.get("dfs", [])
                        st.write("Forecast dataframes structure:", dfs_list)  # Debug
                        if dfs_list:
                            import pandas as pd
                            for i, df_data in enumerate(dfs_list):
                                try:
                                    import pandas as pd
                                    df_forecast = pd.DataFrame(df_data)
                                    if not df_forecast.empty:
                                        st.subheader(f"Forecast Dataframe {i+1}")
                                        st.line_chart(df_forecast)
                                    else:
                                        st.warning(f"Forecast dataframe {i+1} is empty.")
                                except Exception as e:
                                    st.error(f"Error converting dataframe {i+1}: {str(e)}")
                                    st.write("Data structure:", df_data)
                        else:
                            st.warning("No forecast data available.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            # Clear progress displays
            progress_bar.empty()
            status_text.empty()

def create_download_link(report_id):
    """Create a download link for base64-encoded markdown report"""
    download_url = f"{API_URL}/download_report/{report_id}"
    return f'<a href="{download_url}" target="_blank" download="crime_report_{report_id}.md">Download Report with Images</a>'
# Add this helper function in the Helper Functions section
def poll_report_status(report_id: str, status_url: str) -> dict:
    """Simple polling with fixed delay"""
    max_retries = 30  # 30 seconds max wait
    
    for _ in range(max_retries):
        try:
            response = requests.get(f"{API_URL}{status_url}")
            if response.status_code == 200:
                data = response.json()
                if data["status"] in ["completed", "failed"]:
                    return data
            time.sleep(1)  # Simple 1-second delay between checks
        except Exception as e:
            st.error(f"Error checking status: {str(e)}")
            time.sleep(1)
    
    return {"status": "timeout"}

# -------------------------------
# 4) Sidebar Configuration
# -------------------------------
st.sidebar.title("Crime Analysis Assistant")
st.sidebar.markdown("### Search Configuration")

search_type = st.sidebar.radio(
    "Select Search Type",
    ["All Years", "Specific Year Range"],
    key="search_type"
)




if search_type == "Specific Year Range":
    # Replace multiselect with a year range slider
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=1995,
        max_value=2025,
        value=(2020, 2025),  # Default range
        step=1,
        key="year_slider"
    )
    
    # Convert selected years to quarters
    selected_periods = []
    for year in range(year_range[0], year_range[1] + 1):
        for quarter in range(1, 5):
            selected_periods.append(f"{year}q{quarter}")
    
    st.session_state.selected_periods = selected_periods
    st.session_state.start_year = year_range[0]
    st.session_state.end_year = year_range[1]

else:
    selected_periods = ["all"]
    st.session_state.selected_periods = selected_periods
    st.session_state.start_year = None
    st.session_state.end_year = None

# -------------------------------
# LLM Model Configuration
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Configuration")

# Get available models from API

response = requests.get(MODELS_URL)
if response.status_code == 200:
    model_data = response.json()
    available_models = {model: model for model in model_data.get("models", [])}
# Model selection - simplified to avoid unnecessary state changes
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(available_models.keys())[0]

# Create a selection box for model selection without explicit rerun
model_display_name = st.sidebar.selectbox(
    "Select Model",
    options=list(available_models.keys()),
    index=list(available_models.keys()).index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models.keys() else 0,
    key="model_select",
    on_change=lambda: setattr(st.session_state, "selected_model", st.session_state.model_select)
)
# Add model description based on selection
model_descriptions = {
    "Claude 3 Haiku": "Fast, compact model with strong reasoning capabilities (Anthropic)",
    "Claude 3 Sonnet": "Balanced performance with enhanced reasoning (Anthropic)", 
    "Gemini Pro": "Google's advanced model with strong coding abilities (Google)",
    "DeepSeek": "Specialized for code generation and technical tasks (DeepSeek)",
    "Grok": "Conversational model focused on insightful responses (xAI)"
}

with st.sidebar.expander("📝 Model Info"):
    st.markdown(f"**{model_display_name}**")
    st.markdown(model_descriptions.get(model_display_name, "No description available"))

# -------------------------------
# 5) Navigation
# -------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reports" not in st.session_state:
    st.session_state.reports = {}

# Define a callback function for page navigation
def nav_to(page_name):
    st.session_state.current_page = page_name

# Create navigation buttons with callbacks instead of rerun
home_btn = st.sidebar.button("Home", key="nav_Home", on_click=nav_to, args=["Home"], use_container_width=True)
report_btn = st.sidebar.button("Combined Report", key="nav_Report", on_click=nav_to, args=["Combined Report"], use_container_width=True)
map_btn = st.sidebar.button("Map View", key="nav_Map", on_click=nav_to, args=["Map View"], use_container_width=True)
judge_btn = st.sidebar.button("Judge Metrics", key="nav_Judge", on_click=nav_to, args=["Judge Metrics"], use_container_width=True)
about_btn = st.sidebar.button("About", key="nav_About", on_click=nav_to, args=["About"], use_container_width=True)
token_btn = st.sidebar.button("Token Usage", key="nav_Token", on_click=nav_to, args=["Token Usage"], use_container_width=True)

page = st.session_state.current_page

# -------------------------------
# 6) Page Layout
# -------------------------------
if page == "Home":
    st.title("Welcome to the Crime Analysis Assistant")
    st.markdown("""
        This application integrates multiple agents to analyze crime data:
        - **RAG Agent**: Retrieves historical crime reports from our database.
        - **Web Search Agent**: Provides real-time crime statistics via SerpAPI.
        - **Data Agent**: Queries structured crime metrics and displays charts.
    """)
    
    st.subheader("How to use this application:")
    st.markdown("""
    1. **Select cities**: Visit the Map View page and click on cities you want to analyze
    2. **Choose time range**: Use the sidebar to select 'All Years' or a specific year range
    3. **Select model**: Choose the AI model you want to use for analysis
    4. **Generate report**: Go to the Combined Report page and enter your research question
    5. **View metrics**: After generating a report, visit the Judge Metrics page to see quality assessment
    """)

elif page == "Combined Report":
    handle_combined_report()

elif page == "Map View":
    st.title("City Crime Statistics")
    
    # Initialize selected_cities in session state if not present
    if "selected_cities" not in st.session_state:
        st.session_state["selected_cities"] = []

    # Create Folium map with a dark style
    m = folium.Map(
        location=[39.8283, -98.5795], 
        zoom_start=4, 
        tiles="CartoDB dark_matter",   # Dark map style
        attr="CartoDB dark_matter"
    )

    # Add markers for each city
    for city_name, info in cities.items():
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=city_name,
            tooltip=f"Click for {city_name}",
            icon=folium.Icon(icon="fa-map-marker", prefix="fa", color="blue", icon_color="white")
        ).add_to(m)

    # Display the map
    st_map = st_folium(m, width=700, height=500)

    # Handle clicks
    clicked_city_name = st_map.get("last_object_clicked_popup")
    if clicked_city_name:
        if clicked_city_name in cities:
            if clicked_city_name not in st.session_state["selected_cities"]:
                st.session_state["selected_cities"].append(clicked_city_name)
            st.success(f"✅ You selected: {clicked_city_name}")
        else:
            st.warning("⚠️ You clicked on the map but missed a valid marker. Try again.")
    
    # Add a button to clear selections
    if st.session_state["selected_cities"]:
        if st.button("Clear All Selections"):
            st.session_state["selected_cities"] = []
            st.success("✅ All city selections cleared.")


    # Display selected cities
    st.write("---")
    st.subheader("Selected Cities for Analysis")
    if st.session_state["selected_cities"]:
        # Create columns for city info
        cols = st.columns(min(3, len(st.session_state["selected_cities"])))
        
        for i, city in enumerate(st.session_state["selected_cities"]):
            info = cities[city]
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.markdown(f"""
                **{city}**
                - Population: {info['pop']:,}
                - Crime Rate: {info['crime_rate']:.1f}/1000
                - Coordinates: {info['lat']:.4f}, {info['lon']:.4f}
                """)
                
                # Add individual remove button
                if st.button(f"Remove {city}", key=f"remove_{city}"):
                    st.session_state["selected_cities"].remove(city)
                    st.success(f"✅ Removed {city} from selections.")
                    
                    
        # Add a button to navigate to report generation
        st.button("Generate Report with Selected Cities", on_click=nav_to, args=["Combined Report"])
    else:
        st.info("No cities selected yet. Click a marker on the map to add a location.")

elif page == "Judge Metrics":
    st.title("Report Quality Metrics")
    
    if not st.session_state.reports:
        st.warning("No reports have been generated yet.")
        st.button("Generate a Report", on_click=nav_to, args=["Combined Report"])
    else:
        # Get all reports and sort by timestamp
        reports = dict(sorted(
            st.session_state.reports.items(),
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True
        ))
        
        # Display the latest report first
        latest_report_id = next(iter(reports))
        report_data = reports[latest_report_id]
        
        st.subheader("Latest Report Analysis")
        
        # Display metadata
        st.markdown(f"""
            **Question**: {report_data.get('question', 'N/A')}  
            **Generated**: {datetime.fromisoformat(report_data.get('timestamp', '')).strftime('%Y-%m-%d %H:%M:%S')}  
            **Model**: {report_data.get('model', 'N/A')}  
            **Regions**: {', '.join(report_data.get('selected_regions', []))}
        """)
        
        # Extract judge feedback - handle different possible structures
        judge_feedback = report_data.get("evaluation", {})
        
        # Get overall score with proper fallbacks
        score = 0
        if isinstance(judge_feedback, dict):
            # Try different possible paths to get the overall score
            if "overall_score" in judge_feedback:
                score = float(judge_feedback["overall_score"])
            elif isinstance(judge_feedback.get("scores"), dict) and "overall" in judge_feedback["scores"]:
                score = float(judge_feedback["scores"]["overall"])
        else:
            score = float(report_data.get("judge_score", 0))
        
        # Display score with color coding
        if score >= 8:
            st.success(f"### Overall Score: {score:.1f}/10 🌟")
        elif score >= 6:
            st.warning(f"### Overall Score: {score:.1f}/10 ⭐")
        else:
            st.error(f"### Overall Score: {score:.1f}/10 ⚠️")
        
        # Display detailed metrics in two columns with improved handling
        st.subheader("Detailed Metrics")
        
        # Try to get scores from judge_feedback, with multiple fallback options
        scores = {}
        if isinstance(judge_feedback, dict):
            # Option 1: scores are in a 'scores' key
            if isinstance(judge_feedback.get("scores"), dict):
                scores = judge_feedback["scores"]
            # Option 2: scores are directly in judge_feedback (excluding overall_score)
            else:
                scores = {k: v for k, v in judge_feedback.items() 
                         if isinstance(v, (int, float)) and k != "overall_score"}
                         
        # Create metric display
        if scores:
            metrics_cols = st.columns(2)
            for i, (metric, value) in enumerate(scores.items()):
                if isinstance(value, (int, float)):
                    with metrics_cols[i % 2]:
                        # Add color indicators to metrics
                        if value >= 8:
                            st.metric(metric.capitalize(), f"{value:.1f}/10", delta="Good", delta_color="normal")
                        elif value >= 6:
                            st.metric(metric.capitalize(), f"{value:.1f}/10", delta="Average", delta_color="off")
                        else:
                            st.metric(metric.capitalize(), f"{value:.1f}/10", delta="Needs Improvement", delta_color="inverse")
        else:
            st.info("No detailed metrics available for this report.")
        
        # Display improvement suggestions if available
        if isinstance(judge_feedback, dict) and "improvement_suggestions" in judge_feedback:
            suggestions = judge_feedback["improvement_suggestions"]
            if suggestions:
                st.subheader("Improvement Suggestions")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
        
        # Display overall assessment if available
        if isinstance(judge_feedback, dict) and "overall_assessment" in judge_feedback:
            st.subheader("Judge Assessment")
            st.markdown(judge_feedback["overall_assessment"])
        
        # Display feedback history if available (from evaluation data)
        if isinstance(judge_feedback, dict) and "feedback_history" in judge_feedback:
            with st.expander("📜 Feedback History", expanded=False):
                for i, feedback in enumerate(judge_feedback["feedback_history"]):
                    st.markdown(f"**Previous Evaluation #{i+1}**")
                    st.markdown(f"Score: {feedback.get('overall_score', 'N/A')}/10")
                    st.markdown(f"Assessment: {feedback.get('overall_assessment', 'N/A')}")
                    st.markdown("---")
        
        # Show report content in expander
        with st.expander("View Report Content", expanded=False):
            st.markdown(report_data.get("content", ""), unsafe_allow_html=True)
            report_md = create_downloadable_report(report_data)
            st.download_button(
                label="📥 Download Report",
                data=report_md,
                file_name=f"crime_report_{datetime.fromisoformat(report_data.get('timestamp', '')).strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Previous reports section
        if len(reports) > 1:
            st.markdown("---")
            st.subheader("Previous Reports")
            for report_id, data in list(reports.items())[1:]:  # Skip the latest report
                with st.expander(f"Report from {datetime.fromisoformat(data.get('timestamp', '')).strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.markdown(f"""
                        **Question**: {data.get('question', 'N/A')}  
                        **Score**: {data.get('judge_score', 0)}/10  
                        **Model**: {data.get('model', 'N/A')}  
                        **Regions**: {', '.join(data.get('selected_regions', []))}
                    """)
                    
                    # Add a view button that expands to show full content
                    if st.button(f"View Full Report", key=f"view_report_{report_id}"):
                        st.markdown(data.get("content", ""))
                        
                        # Also show judge feedback for this report
                        prev_judge_feedback = data.get("judge_feedback", {})
                        if prev_judge_feedback:
                            st.subheader("Judge Feedback")
                            st.json(prev_judge_feedback)

elif page == "Token Usage":
    st.title("Token Usage Summary")
    if st.session_state.reports:
        total_tokens = 0
        total_cost = 0
        
        for rep_id, rep_data in st.session_state.reports.items():
            st.subheader(f"Report Generated: {datetime.fromisoformat(rep_data.get('timestamp', '')).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get token usage from the report data
            token_summary = rep_data.get("token_usage_summary", {})
            if token_summary:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tokens", f"{token_summary.get('total_tokens', 0):,}")
                    total_tokens += token_summary.get('total_tokens', 0)
                with col2:
                    st.metric("Total Cost", f"${token_summary.get('total_cost', 0):.4f}")
                    total_cost += token_summary.get('total_cost', 0)
                
                # Display per-node breakdown
                if node_usage := token_summary.get("by_node"):
                    with st.expander("🔍 Token Usage by Component"):
                        for node, usage in node_usage.items():
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric(f"{node} Tokens", f"{usage.get('tokens', 0):,}")
                            with cols[1]:
                                st.metric(f"{node} Cost", f"${usage.get('cost', 0):.4f}")
        
        # Display overall statistics
        st.markdown("---")
        st.subheader("💰 Overall Usage Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens (All Reports)", f"{total_tokens:,}")
        with col2:
            st.metric("Total Cost (All Reports)", f"${total_cost:.4f}")

elif page == "About":
    st.title("About Crime Analysis Assistant")
    st.markdown("""
        **Crime Analysis Assistant** integrates:
        - **RAG Agent**: Uses database with metadata filtering to retrieve historical crime reports.
        - **Web Search Agent**: Uses SerpAPI for real-time crime statistics.
        - **Data Agent**: Connects to databases for crime metrics and visualization.
    """)
    
    st.subheader("System Architecture")
    st.markdown("""
    This system uses a multi-agent approach to analyze crime data:
    
    1. **Frontend**: Streamlit application for user interaction
    2. **Backend**: FastAPI server running the analysis pipeline
    3. **Agents**: Multiple specialized AI agents that work together:
       - RAG Agent: Retrieves and processes historical data
       - Web Search Agent: Gathers current information from the web
       - Data Analysis Agent: Performs statistical analysis on crime data
       - Comparison Agent: Compares trends across different regions
       - Judge Agent: Evaluates report quality
       
    4. **LLM Integration**: Uses multiple language models (Claude, Gemini, etc.) for analysis
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. **Select cities** on the Map View
    2. **Choose time range** in the sidebar
    3. **Enter your query** on the Combined Report page
    4. View the generated report
    5. Check quality metrics on the Judge Metrics page
    """)

# Inject FontAwesome for map icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
""", unsafe_allow_html=True)

# Then load your custom CSS
with open("styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)