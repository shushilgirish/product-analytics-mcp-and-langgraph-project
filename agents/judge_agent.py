from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import json
import traceback
from datetime import datetime
from typing import Dict, List
from agents.llmselection import LLMSelector as llmselection

class JudgeAgent:
    def __init__(self, model_type: str = None):
        self.model_type = model_type
        self.llm = llmselection.get_llm(self.model_type)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="evaluation_history",
            return_messages=True,
            input_key="report",
            output_key="evaluation"
        )
        
        # Initialize tools
        self.tools = [
            Tool(
                name="evaluate_accuracy",
                func=self._evaluate_accuracy,
                description="Evaluate factual accuracy and correctness"
            ),
            Tool(
                name="evaluate_completeness",
                func=self._evaluate_completeness,
                description="Evaluate report comprehensiveness"
            ),
            Tool(
                name="compare_with_previous",
                func=self._compare_with_previous,
                description="Compare with previous report evaluations"
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        # Store feedback history
        self.feedback_history = []
    
    def evaluate(self, context):
        """
        Evaluate a crime report for quality and completeness.
        
        Args:
            context (dict): Evaluation context including report and parameters
        
        Returns:
            dict: Evaluation results including scores and feedback
        """
        try:
            # Extract report data
            report = context.get('report', {})
            
            # Initialize scores
            scores = {
                "completeness": 0,
                "accuracy": 0,
                "usefulness": 0,
                "clarity": 0
            }
            
            # Check if report has basic structure
            if not report or not isinstance(report, dict):
                print("Warning: Invalid report structure")
                return {
                    "overall_assessment": "Report evaluation failed due to invalid structure",
                    "overall_score": 5,
                    "scores": scores,
                    "improvement_suggestions": ["Fix report structure"]
                }
            
            # Prepare report for evaluation
            report_title = report.get("title", "Untitled Report")
            sections = report.get("sections", [])
            section_titles = [s.get("title", "Untitled Section") for s in sections]
            
            # Create a simple evaluation based on report structure
            section_count = len(sections)
            has_exec_summary = any("executive" in s.get("title", "").lower() for s in sections)
            has_methodology = any("methodology" in s.get("title", "").lower() for s in sections)
            has_analysis = any("analysis" in s.get("title", "").lower() for s in sections)
            has_recommendations = any("recommend" in s.get("title", "").lower() for s in sections)
            
            # Calculate completeness score
            completeness = 0
            if section_count >= 8:
                completeness = 10
            elif section_count >= 6:
                completeness = 8
            elif section_count >= 4:
                completeness = 6
            else:
                completeness = 4
                
            # Adjust for key sections
            if has_exec_summary:
                completeness = min(10, completeness + 1)
            if has_methodology:
                completeness = min(10, completeness + 1)
            if has_analysis:
                completeness = min(10, completeness + 1)
            if has_recommendations:
                completeness = min(10, completeness + 1)
                
            scores["completeness"] = completeness
            
            # For other metrics, assign reasonable defaults
            scores["accuracy"] = 7  # Hard to evaluate without ground truth
            scores["usefulness"] = 8 if has_recommendations else 6
            scores["clarity"] = 7  # Default assumption
            
            # Calculate overall score
            overall_score = round(sum(scores.values()) / len(scores))
            
            # Generate improvement suggestions
            suggestions = []
            if not has_exec_summary:
                suggestions.append("Add an Executive Summary section")
            if not has_methodology:
                suggestions.append("Include a Methodology section")
            if not has_analysis:
                suggestions.append("Provide more detailed analysis of crime data")
            if not has_recommendations:
                suggestions.append("Include actionable recommendations")
                
            if not suggestions:
                suggestions.append("Consider adding more visualizations to illustrate key points")
                
            # Generate overall assessment
            assessment = f"Report '{report_title}' evaluated with an overall score of {overall_score}/10. "
            if overall_score >= 8:
                assessment += "This is a high-quality report with comprehensive coverage of crime analysis."
            elif overall_score >= 6:
                assessment += "This is a good report that meets basic requirements but could be improved."
            else:
                assessment += "This report needs significant improvements to meet quality standards."
            
            # Store feedback without adding a second state parameter
            self.feedback_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "report_title": report_title,
                "overall_score": overall_score,
                "improvement_suggestions": suggestions
            })
            
            # Return evaluation results
            return {
                "overall_assessment": assessment,
                "overall_score": overall_score,
                "scores": scores,
                "improvement_suggestions": suggestions
            }
        
        except Exception as e:
            print(f"JudgeAgent evaluation error: {str(e)}")
            traceback.print_exc()
            return {
                "overall_assessment": f"Evaluation failed due to error: {str(e)}",
                "overall_score": 5,
                "scores": {
                    "completeness": 5,
                    "accuracy": 5,
                    "usefulness": 5,
                    "clarity": 5
                },
                "improvement_suggestions": ["Fix evaluation process"]
            }    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response when evaluation fails."""
        return {
            "status": "error",
            "error": error_message,
            "overall_score": 5,  # Default middle score
            "scores": {
                "completeness": 5,
                "accuracy": 5, 
                "usefulness": 5,
                "clarity": 5,
                "overall": 5
            },
            "feedback": {
                "strengths": ["Unable to evaluate due to error"],
                "weaknesses": ["Error during evaluation process"],
                "improvements": [f"Technical error: {error_message}"]
            },
            "overall_assessment": f"Report evaluation failed due to technical error: {error_message}"
        }

    def _get_relevant_feedback(self, state: Dict) -> List[Dict]:
        """Retrieve relevant feedback from past evaluations."""
        try:
            # Return most recent feedback entries if available
            return self.feedback_history[-3:] if hasattr(self, 'feedback_history') and self.feedback_history else []
        except Exception as e:
            print(f"Error retrieving feedback: {str(e)}")
            return []
        
    def _evaluate_completeness(self, report_data: Dict) -> Dict:
        """Evaluate completeness of the report."""
        required_sections = [
            "executive_summary",
            "methodology",
            "analysis",
            "recommendations"
        ]
        
        completeness = {
            "score": 0,
            "missing_sections": [],
            "feedback": []
        }
        
        sections = report_data.get("sections", [])
        section_titles = [s.get("title", "").lower() for s in sections]
        
        for required in required_sections:
            if not any(required in title for title in section_titles):
                completeness["missing_sections"].append(required)
        
        completeness["score"] = 10 - (len(completeness["missing_sections"]) * 2)
        return completeness

    def _compare_with_previous(self, current_evaluation: Dict) -> List[str]:
        """Compare with previous evaluations from memory."""
        if not self.feedback_history:
            return ["No previous evaluations available"]
            
        previous = self.feedback_history[-1] if self.feedback_history else None
        if not previous:
            return ["First evaluation"]
            
        improvements = []
        current_score = current_evaluation.get("scores", {}).get("overall", 0)
        previous_score = previous.get("evaluation", {}).get("scores", {}).get("overall", 0)
        
        if current_score > previous_score:
            improvements.append(f"Overall score improved by {current_score - previous_score} points")
        elif current_score < previous_score:
            improvements.append(f"Overall score decreased by {previous_score - current_score} points")
            
        return improvements
    def _create_evaluation_prompt(self,context: Dict) -> str:
        """Create evaluation prompt template"""
        report_content = json.dumps(context.get("report", {}), indent=2)[:1500]  # Truncate to avoid token limits
    
        eval_prompt = f"""
        Evaluate this crime analysis report based on:
        1. Completeness (1-10)
        2. Accuracy (1-10)
        3. Usefulness (1-10)
        4. Clarity (1-10)

        Report Content:
        {report_content}

        Regions: {", ".join(context.get("regions", []))}
        Time Period: {context.get("time_period", "")}

        Provide scores and detailed feedback.
        """
        return eval_prompt
    
    def _run_evaluation(self, context: Dict) -> Dict:
        """Run the actual evaluation using the agent."""
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(context)
        
        # Get evaluation from agent
        try:
            response = self.agent.run(eval_prompt)
        except Exception as e:
            print(f"Agent evaluation error: {str(e)}")
            response = f"Error evaluating report: {str(e)}"
        
        # Process and structure the response
        try:
            evaluation = json.loads(response)
        except:
            evaluation = {
                "status": "error",
                "error": "Failed to parse evaluation response",
                "overall_score": 5,
                "scores": {
                    "completeness": 5,
                    "accuracy": 5, 
                    "usefulness": 5,
                    "clarity": 5,
                    "overall": 5
                },
                "feedback": {
                    "strengths": ["Unable to evaluate due to error"],
                    "weaknesses": ["Error during evaluation process"],
                    "improvements": [f"Technical error: {response}"]
                },
                "overall_assessment": f"Report evaluation failed due to technical error: {response}",
                "overall_score": 5
            }
        
        return evaluation
    def _evaluate_accuracy(self, report_data: Dict) -> Dict:
        """Evaluate factual accuracy of the report."""
        try:
            sections = report_data.get("sections", [])
            evaluation = {
                "score": 0,
                "feedback": [],
                "improvements": []
            }
            
            # Check for data citations and sources
            for section in sections:
                content = section.get("content", "")
                if content and isinstance(content, str):
                    # Score based on data references
                    data_references = len([line for line in content.split('\n') if any(term in line.lower() for term in ["data shows", "statistics indicate", "according to", "analysis reveals"])])
                    evaluation["score"] += min(data_references, 3)  # Max 3 points per section
                    
            # Normalize score to 1-10 range
            evaluation["score"] = min(max(evaluation["score"], 1), 10)
            return evaluation
        except Exception as e:
            return {"score": 5, "error": str(e)}
    
    def _store_feedback(self, evaluation: Dict, state: Dict) -> None:
        """Store feedback for future reference."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": state.get("question", ""),
            "regions": state.get("selected_regions", []),
            "model_type": state.get("model_type", ""),
            "evaluation": evaluation,
            "improvements_suggested": evaluation.get("improvements", {})
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update memory with structured feedback
        self.memory.save_context(
            {"feedback": json.dumps(feedback_entry)},
            {"stored": "Feedback stored successfully"}
        )
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get improvement suggestions based on feedback history."""
        if not self.feedback_history:
            return []
            
        # Analyze feedback history for common improvement areas
        improvements = []
        for entry in self.feedback_history[-5:]:  # Look at last 5 evaluations
            for improvement in entry.get("improvements_suggested", {}).values():
                improvements.append(improvement)
                
        return list(set(improvements))  # Remove duplicates