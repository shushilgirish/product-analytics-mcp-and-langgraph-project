import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llmselection import LLMSelector as llmselection
import pandas as pd
class ForecastAgent:
    def __init__(self, model_type: str = "Gemini Pro"):
        self.model_type = model_type
        self.llm = llmselection.get_llm(model_type)
        
    def generate_forecast_data(self, snowflake_data: Dict, rag_data: Dict, 
                             web_data: Dict, comparison_data: Dict,
                             regions: List[str]) -> Dict[str, Any]:
        """Generate enhanced forecasts using multiple data sources."""
        try:
            # Extract historical trends from Snowflake
            yearly_trends = snowflake_data.get("statistics", {}).get("incident_analysis", {}).get("yearly_trends", {})
            df_historical = pd.DataFrame(yearly_trends)
            
            # Get additional insights
            rag_insights = rag_data.get("insights", "")
            web_trends = web_data.get("markdown_report", "")
            comparison_insights = comparison_data.get("comparison", "")
            
            # Create baseline forecasts
            forecasts = {}
            last_year = df_historical.index.max()
            forecast_years = range(last_year + 1, last_year + 6)
            
            for region in regions:
                # Get region-specific data
                region_data = df_historical.copy()
                
                # Calculate weighted trends using multiple factors
                recent_trend = self._calculate_weighted_trend(
                    historical_data=region_data,
                    rag_insights=rag_insights,
                    web_trends=web_trends,
                    comparison_insights=comparison_insights,
                    region=region
                )
                
                # Generate forecasted values
                forecast_values = []
                last_value = region_data.iloc[-1]
                
                for year in forecast_years:
                    next_value = last_value * (1 + recent_trend)
                    forecast_values.append(next_value)
                    last_value = next_value
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame(
                    forecast_values,
                    index=forecast_years,
                    columns=df_historical.columns
                )
                
                # Add confidence intervals
                confidence_intervals = self._calculate_confidence_intervals(
                    historical=region_data,
                    forecast=forecast_df,
                    region=region
                )
                
                # Combine all data
                combined_df = pd.concat([df_historical, forecast_df])
                combined_df['forecast_type'] = ['historical'] * len(df_historical) + ['forecast'] * len(forecast_df)
                
                forecasts[region] = {
                    'historical': df_historical,
                    'forecast': forecast_df,
                    'combined': combined_df,
                    'confidence_intervals': confidence_intervals,
                    'trend_analysis': {
                        'recent_trend': recent_trend,
                        'seasonality': self._detect_seasonality(region_data),
                        'confidence_score': self._calculate_confidence_score(region_data)
                    }
                }
            
            return {
                "status": "success",
                "forecasts": forecasts,
                "metadata": {
                    "regions": regions,
                    "forecast_years": list(forecast_years),
                    "data_sources": ["Snowflake", "RAG", "Web Search", "Comparison Analysis"],
                    "generated_at": datetime.now().isoformat(),
                    "model": self.model_type
                }
            }
            
        except Exception as e:
            print(f"Error generating enhanced forecasts: {str(e)}")
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}

    def _calculate_weighted_trend(self, historical_data: pd.DataFrame, 
                                rag_insights: str, web_trends: str,
                                comparison_insights: str, region: str) -> float:
        """Calculate weighted trend using multiple data sources."""
        # Base trend from historical data (last 3 years)
        base_trend = historical_data.iloc[-3:].mean().pct_change().mean()
        
        # Analyze text insights for trend modifiers
        trend_modifiers = {
            'increasing': 1.1,
            'decreasing': 0.9,
            'stable': 1.0,
            'significant': 1.2,
            'slight': 1.05
        }
        
        modifier = 1.0
        combined_text = f"{rag_insights} {web_trends} {comparison_insights}".lower()
        
        for term, value in trend_modifiers.items():
            if term in combined_text:
                modifier *= value
        
        return base_trend * modifier

    def _calculate_confidence_intervals(self, historical: pd.DataFrame, 
                                     forecast: pd.DataFrame, region: str) -> Dict:
        """Calculate confidence intervals for forecasts."""
        std_dev = historical.std()
        return {
            'lower_bound': forecast - (1.96 * std_dev),
            'upper_bound': forecast + (1.96 * std_dev)
        }

    def _detect_seasonality(self, data: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in historical data."""
        try:
            # Simple seasonality detection using autocorrelation
            autocorr = data.apply(lambda x: x.autocorr())
            return {
                'has_seasonality': bool(autocorr.mean() > 0.7),
                'strength': float(autocorr.mean())
            }
        except:
            return {'has_seasonality': False, 'strength': 0.0}

    def _calculate_confidence_score(self, data: pd.DataFrame) -> float:
        """Calculate confidence score for predictions."""
        try:
            # Factors affecting confidence:
            # 1. Data consistency
            consistency = 1 - data.isnull().sum().mean()
            # 2. Trend stability
            stability = 1 - abs(data.pct_change().std().mean())
            # 3. Data quantity
            quantity = min(len(data) / 10, 1)  # Max score at 10 years of data
            
            return (consistency + stability + quantity) / 3
        except:
            return 0.5