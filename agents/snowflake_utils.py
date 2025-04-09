# agents/snowflake_agent.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
import warnings
from io import BytesIO
import base64
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, field_validator
from datetime import datetime, timezone
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llmselection import LLMSelector as llmselection

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv(override=True)

class CrimeReportRequest(BaseModel):
    """Request model for generating the crime report"""
    question: str
    search_mode: str = "all_years"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    model_type: Optional[str] = "Gemini Pro"
    selected_regions: List[str]  

    @field_validator('search_mode')
    def validate_search_mode(cls, v):
        if v not in ["all_years", "specific_range"]:
            raise ValueError('search_mode must be either "all_years" or "specific_range"')
        return v

    @field_validator('selected_regions')
    def validate_selected_regions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one region must be selected')
        return v

    
    
class CrimeDataAnalyzer:
    def __init__(self, engine, llm):
        self.engine = engine
        self.llm = llm
        self.current_user = "user"
        self.current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self._initialize_agent()
  
    def _initialize_agent(self):
        """Initialize or reinitialize the LangChain agent with current LLM."""
        # Create LangChain tool
        crime_analysis_tool = Tool(
            name="crime_analysis_context",
            description="Analyze crime data context",
            func=lambda x: x  # Simple pass-through function
        )
        
        # Initialize LangChain agent
        self.agent = initialize_agent(
            tools=[crime_analysis_tool],
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=2,
            early_stopping_method="generate"
        )    
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute the provided SQL query using the engine and return a DataFrame."""
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            raise e
            
    
    def create_base_query(self, request: CrimeReportRequest) -> str:
        """Create base SQL query with filters based on CrimeReportRequest."""
        base_query = """
            SELECT 
                EXTRACT(YEAR FROM date) as year,
                incident,
                city,
                zipcode,
                SUM(value) as incident_count
            FROM CLEANED_CRIME_DATASET
            WHERE 1=1
        """
        
        # Add multi-region filter
        regions = "','".join(request.selected_regions)
        base_query += f" AND city IN ('{regions}')"
        
        # Add year range filter based on search_mode
        if request.search_mode == "specific_range":
            base_query += f" AND EXTRACT(YEAR FROM date) BETWEEN {request.start_year} AND {request.end_year}"
        
        # Group by year, incident, city, and zipcode
        base_query += """
            GROUP BY year, incident, city, zipcode
            ORDER BY year, incident, city, zipcode
        """
            
        return base_query
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics from DataFrame with incident analysis."""
        stats = {}
        
        # Overall statistics
        stats["total_incidents"] = df['INCIDENT_COUNT'].sum()
        stats["years_range"] = sorted(df['YEAR'].unique())
        stats["num_years"] = len(stats["years_range"])
        stats["yearly_avg"] = stats["total_incidents"] / stats["num_years"] if stats["num_years"] > 0 else 0
        
        # Incident type analysis
        incident_stats = df.groupby('INCIDENT').agg({
            'INCIDENT_COUNT': ['sum', 'mean', 'std']
        }).round(2)
        incident_stats.columns = ['total', 'mean', 'std']
        stats["incident_analysis"] = incident_stats.sort_values('total', ascending=False)
        
        # Yearly trends per incident
        yearly_trends = df.pivot_table(
            index='YEAR',
            columns='INCIDENT',
            values='INCIDENT_COUNT',
            aggfunc='sum'
        ).fillna(0)
        stats["yearly_trends"] = yearly_trends
        
        # Calculate year-over-year growth for each incident type
        yoy_growth = yearly_trends.pct_change() * 100
        stats["yoy_growth"] = yoy_growth
        
        # Location-based analysis if multiple regions
        if len(df['CITY'].unique()) > 1:
            location_stats = df.groupby(['CITY', 'INCIDENT'])['INCIDENT_COUNT'].sum().unstack()
            stats["location_comparison"] = location_stats
        
        return stats
    
    def analyze_crime_data(self, request: CrimeReportRequest) -> dict:
        """Main method to analyze crime data based on CrimeReportRequest."""
        try:
            print(f"Starting analysis for request: {request}")
            
            # Reinitialize LLM with the requested model type
            if request.model_type:
                print(f"Initializing model: {request.model_type}")
                # Get a new LLM instance based on the requested model type
                self.llm = llmselection.get_llm(request.model_type)
                self._initialize_agent()
            
            # Execute query and get data
            query = self.create_base_query(request)
            print(f"Executing query: {query}")
            df = self.execute_query(query)
            df.columns = df.columns.str.upper()
            
            if df.empty:
                raise ValueError("No data returned for the specified filters.")
            
            print(f"Retrieved {len(df)} rows of data")
            print("DataFrame columns:", df.columns)
            print("Sample data:\n", df.head())
            
            # Calculate statistics
            stats = self.calculate_statistics(df)
            print("Statistics calculated successfully")
            
            # Generate visualization
            chart_path = self.generate_visualization(df, request)
            print(f"Visualization generated and saved to {chart_path}")
            
            # Prepare analysis context
            context = self.create_analysis_context(request, stats, df)
            
            # Get LLM analysis
            analysis = self._get_llm_analysis(context)
            print("LLM analysis completed")
            
            # Create and return response
            response = self.create_response(request, stats, chart_path, analysis)
            print("Analysis completed successfully")
            return response
            
        except Exception as e:
            print(f"Error in analyze_crime_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                "request_info": {
                    "selected_regions": request.selected_regions,
                    "search_mode": request.search_mode,
                    "year_range": f"{request.start_year}-{request.end_year}" if request.search_mode == "specific_range" else "all years"
                }
            }
        
    def _get_llm_analysis(self, context: str) -> str:
        try:
            # Use the agent to get the response
            response = self.agent.run(f"Analyze this crime data: {context}")
            return response

        except Exception as e:
            print(f"Error in LLM analysis: {str(e)}")
            return f"""
            Analysis could not be generated due to an error. 
            Please review the statistical data and visualizations provided.
            
            Key metrics available:
            - Total incidents and yearly averages
            - Incident type distribution
            - Year-over-year trends
            - Geographic distribution (if multiple regions)
            
            Error details: {str(e)}
            """

    def generate_visualization(self, df: pd.DataFrame, request: CrimeReportRequest) -> dict:
        """Generate separate visualizations for different aspects of crime data."""
        paths = {}
        comparison_paths = {}
        
        try:
            plt.style.use('default')
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # 1. Total Crime Incidents by City Over Time
            plt.figure(figsize=(15, 8))
            
            for city in request.selected_regions:
                city_data = df[df['CITY'] == city]
                # Sum all incident counts for each year, rather than looking for 'all incidents'
                all_incidents = city_data.groupby('YEAR')['INCIDENT_COUNT'].sum()
                
                plt.plot(all_incidents.index, all_incidents.values,
                        marker='o',
                        label=city,
                        linewidth=2,
                        markersize=8)
                
                # If you still want numeric labels on this chart, uncomment the annotation below.
                # for x, y in zip(all_incidents.index, all_incidents.values):
                #     plt.annotate(f'{int(y):,}',
                #                  xy=(x, y),
                #                  xytext=(0, 10),
                #                  textcoords='offset points',
                #                  ha='center',
                #                  fontsize=9)

            plt.title('Total Crime Incidents by City Over Time', fontsize=16, pad=20)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Incidents', fontsize=14)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(fontsize=12, loc='upper left')
            plt.xticks(rotation=45)
            plt.margins(y=0.1)
            
            plt.tight_layout()
            all_incidents_path = "all_incidents_trend.png"
            plt.savefig(all_incidents_path, dpi=150, bbox_inches='tight')
            plt.close()
            paths['all_incidents_trend'] = all_incidents_path

            # 2. Top 5 Incidents by City (Remove numeric labels on lines)
            for city in request.selected_regions:
                plt.figure(figsize=(15, 8))
                
                # Filter data for current city
                city_df = df[df['CITY'] == city]
                
                # Get top 5 incidents (excluding 'all incidents' if it exists in your data)
                # If you do not have 'all incidents' at all, just remove that filter
                top_5_incidents = (
                    city_df[city_df['INCIDENT'] != 'all incidents']
                    .groupby('INCIDENT')['INCIDENT_COUNT']
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                    .index
                )
                
                # Create separate trends for each incident
                for idx, incident in enumerate(top_5_incidents):
                    incident_data = city_df[city_df['INCIDENT'] == incident].groupby('YEAR')['INCIDENT_COUNT'].sum()
                    
                    plt.plot(incident_data.index, 
                            incident_data.values,
                            marker='o',
                            label=incident.strip(),
                            color=colors[idx],
                            linewidth=2,
                            markersize=6)
                    
                    # *** Remove the annotations here ***
                    # for x, y in zip(incident_data.index, incident_data.values):
                    #     if y > 0:
                    #         y_offset = 10 if idx % 2 == 0 else -15
                    #         plt.annotate(f'{int(y):,}',
                    #                      xy=(x, y),
                    #                      xytext=(0, y_offset),
                    #                      textcoords='offset points',
                    #                      ha='center',
                    #                      va='bottom' if y_offset > 0 else 'top',
                    #                      fontsize=8)

                plt.title(f'Top 5 Crime Types - {city}', fontsize=16, pad=20)
                plt.xlabel('Year', fontsize=14)
                plt.ylabel('Number of Incidents', fontsize=14)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                
                # Add more space for labels
                plt.margins(y=0.2)
                
                city_incidents_path = f'top5_incidents_{city.replace(" ", "_")}.png'
                plt.tight_layout()
                plt.savefig(city_incidents_path, dpi=150, bbox_inches='tight')
                plt.close()
                comparison_paths[f'top5_incidents_{city.replace(" ", "_")}'] = city_incidents_path

            # 3. Horizontal Yearly Distribution
            plt.figure(figsize=(12, 15))
            
            # Prepare data
            yearly_city_totals = df.pivot_table(
                index='YEAR',
                columns='CITY',
                values='INCIDENT_COUNT',
                aggfunc='sum'
            ).fillna(0)
            
            # Create horizontal stacked bars
            yearly_city_totals.plot(
                kind='barh',
                stacked=True,
                color=colors[:len(request.selected_regions)]
            )
            
            plt.title('Yearly Crime Distribution by City', fontsize=16, pad=20)
            plt.xlabel('Number of Incidents', fontsize=14)
            plt.ylabel('Year', fontsize=14)
            
            plt.legend(title='Cities', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Add total values at the end of each bar (optional; remove if you don't want these)
            totals = yearly_city_totals.sum(axis=1)
            for i, total in enumerate(totals):
                plt.text(total, i, f'  {int(total):,}',
                        va='center',
                        fontsize=9)
            
            plt.tight_layout()
            distribution_path = "yearly_distribution.png"
            plt.savefig(distribution_path, dpi=150, bbox_inches='tight')
            plt.close()
            comparison_paths['yearly_distribution'] = distribution_path

            return {
                "status": "success",
                "paths": paths,
                "comparison_paths": comparison_paths,
                "metadata": {
                    "generated_at": self.current_time,
                    "generated_by": self.current_user,
                    "selected_regions": request.selected_regions,
                    "year_range": (
                        f"{request.start_year}-{request.end_year}" 
                        if request.search_mode == "specific_range" 
                        else "all years"
                    )
                }
            }

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            print("\nDebug Information:")
            print("DataFrame columns:", df.columns.tolist())
            print("DataFrame sample:\n", df.head())
            print("DataFrame info:")
            print(df.info())
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": self.current_time,
                "paths": {},
                "metadata": {
                    "generated_at": self.current_time,
                    "generated_by": self.current_user,
                    "error_details": str(e)
                }
            }

    
    def create_analysis_context(self, request: CrimeReportRequest, stats: dict, df: pd.DataFrame) -> str:
        """Create detailed context for LLM analysis focusing on incidents."""
        try: 
            regions = ', '.join(request.selected_regions)
            
            # Calculate incident statistics without std
            incident_stats = df.groupby('INCIDENT').agg({
                'INCIDENT_COUNT': ['sum', 'mean']
            }).round(2)
            incident_stats.columns = ['total', 'mean']
            
            # Sort by total count and get top incidents
            top_incidents = incident_stats.sort_values(('total'), ascending=False).head()

            context = f"""
            Analyzing crime incidents for {regions} from {min(stats['years_range'])} to {max(stats['years_range'])}:

            Overall Statistics:
            - Total Reported Incidents: {stats['total_incidents']:,}
            - Yearly Average: {stats['yearly_avg']:.2f}
            - Number of Years Analyzed: {stats['num_years']}

            Top 5 Incident Types by Total Count:
            {top_incidents.to_string()}

            Yearly Growth Rates (Year-over-Year % Change):
            {stats['yoy_growth'].mean().sort_values(ascending=False).head().to_string()}

            User Question: {request.question}

            Please provide a comprehensive analysis including:
            1. Direct answer to the user's question
            2. Major trends in different types of incidents
            3. Notable changes or patterns over the years
            4. Comparison between different incident types
            5. Any concerning trends or positive developments
            6. Location-based insights (comparing the selected regions)
            7. Recommendations based on the data
            """
            
            return context

        except Exception as e:
            print(f"Error creating analysis context: {str(e)}")
            return f"""
            Analyzing crime data for {', '.join(request.selected_regions)}:
            
            Total Incidents: {df['INCIDENT_COUNT'].sum():,}
            Years Analyzed: {df['YEAR'].min()} to {df['YEAR'].max()}
            
            User Question: {request.question}
            
            Please analyze the available data to address:
            1. Overall crime trends
            2. Comparison between regions
            3. Key patterns and changes
            4. Recommendations based on the data
            """
    
    def create_response(self, request: CrimeReportRequest, stats: dict, visualization_metadata: dict, analysis: str) -> dict:
        """Create detailed response dictionary with incident analysis."""
        return {
            "request_info": {
                "selected_regions": request.selected_regions,
                "search_mode": request.search_mode,
                "year_range": f"{request.start_year}-{request.end_year}" if request.search_mode == "specific_range" else "all years",
                "model_used": request.model_type
            },
            "statistics": {
                "total_incidents": stats['total_incidents'],
                "yearly_average": stats['yearly_avg'],
                "years_analyzed": stats['years_range'],
                "incident_analysis": {
                    "top_incidents": stats['incident_analysis'].head().to_dict(),
                    "yearly_trends": stats['yearly_trends'].to_dict(),
                    "growth_rates": stats['yoy_growth'].mean().to_dict()
                }
            },
            "visualizations": visualization_metadata,
            "analysis": analysis,
            "status": "success"
        }

def initialize_connections(model_type=None):
    """Initialize database and LLM connections."""
    try:
        # Get Snowflake credentials from environment variables
        user = os.getenv('SNOWFLAKE_USER')
        password = os.getenv('SNOWFLAKE_PASSWORD')
        account = os.getenv('SNOWFLAKE_ACCOUNT')
        database = os.getenv('SNOWFLAKE_DATABASE')
        schema = os.getenv('SNOWFLAKE_SCHEMA')
        warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')

        # Verify that all required environment variables are set
        if not all([user, password, account]):
            print("Error: Missing required Snowflake credentials in .env file")
            raise ValueError("Missing Snowflake credentials")

        # Create engine with database, schema and warehouse if available
        connection_string = f'snowflake://{user}:{password}@{account}/'
        if database and schema and warehouse:
            connection_string += f'{database}/{schema}?warehouse={warehouse}'
        
        engine = create_engine(connection_string)
        
        llm = llmselection.get_llm(model_type)
        
        return engine, llm
    except Exception as e:
        print(f"Error initializing connections: {str(e)}")
        raise

if __name__ == "__main__":
    import json
    try:
        # Initialize connections and create analyzer instance
        print("Initializing connections...")
        engine, llm = initialize_connections()
        analyzer = CrimeDataAnalyzer(engine, llm)
        
        # Test with all years
        all_years_request = CrimeReportRequest(
            question="What are the major crime trends in Chicago, New York and Los Angeles across all available years?",
            search_mode="all_years",
            selected_regions=["Chicago", "New York", "Los Angeles"],
            model_type="Gemini Pro"
        )
        
        print("\nRunning analysis for all available years...")
        result = analyzer.analyze_crime_data(all_years_request)
        
        if result["status"] == "success":
            print(f"""
Crime Analysis Report
====================
Regions Analyzed: {', '.join(result['request_info']['selected_regions'])}
Time Period: {result['request_info']['year_range']}
Model Used: {result['request_info']['model_used']}

Statistics:
- Total Incidents: {result['statistics']['total_incidents']:,}
- Yearly Average: {result['statistics']['yearly_average']:.2f}
- Years Analyzed: {', '.join(map(str, result['statistics']['years_analyzed']))}

Visualizations Generated:
{json.dumps(result['visualizations']['paths'], indent=2)}

Analysis:
{result['analysis']}

Generated at: {result['visualizations']['metadata']['generated_at']}
Generated by: {result['visualizations']['metadata']['generated_by']}
""")
        else:
            print(f"""
Analysis Error Report
===================
Status: {result['status']}
Error: {result.get('error', 'Unknown error')}
Timestamp: {result.get('timestamp', 'Unknown')}
""")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()