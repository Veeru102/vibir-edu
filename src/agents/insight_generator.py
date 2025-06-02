from crewai import Agent, Task, Crew
from typing import List, Dict
import json
import re
from ..models.data_models import Insight, ForecastResult, StrategicGoal, BudgetDelta
from langchain_openai import ChatOpenAI

class InsightGenerator:
    def __init__(self):
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        self.insight_agent = Agent(
            role='Budget Insight Analyst',
            goal='Generate meaningful insights from budget forecasts and changes',
            backstory="""You are an experienced budget analyst with expertise in K-12 education finance.
            You excel at identifying patterns, trends, and potential impacts of budget changes on educational outcomes.
            You have a deep understanding of K-12 education funding, including federal, state, and local sources.
            You can analyze the impact of budget changes on student outcomes, teacher effectiveness, and program quality.
            You ALWAYS provide your analysis in the exact format specified, with clear sections for each category.""",
            verbose=True,
            llm=llm,
            allow_delegation=False
        )

    def _parse_llm_output(self, output: str) -> List[Dict]:
        """Parse the LLM output into structured insights."""
        insights = []
        
        # First try to parse as JSON if it's a JSON string
        try:
            parsed_json = json.loads(output)
            if isinstance(parsed_json, list):
                # If it's already a list of insights, return it
                return parsed_json
            elif isinstance(parsed_json, dict):
                # If it's a single insight in dict format, convert to list
                return [parsed_json]
        except json.JSONDecodeError:
            # If not JSON, proceed with regex parsing
            pass
        
        # Split the output into sections for each category
        sections = re.split(r'\n(?=Category:)', output)
        
        for section in sections:
            if not section.strip():
                continue
                
            try:
                # Extract category
                category_match = re.search(r'Category:\s*(.+?)(?=\n|$)', section)
                if not category_match:
                    continue
                category = category_match.group(1).strip()
                
                # Extract insight
                insight_match = re.search(r'Insight:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                insight = insight_match.group(1).strip() if insight_match else ""
                
                # Extract impact
                impact_match = re.search(r'Impact:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                impact = impact_match.group(1).strip() if impact_match else ""
                
                # Extract recommendation
                rec_match = re.search(r'Recommendation:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                recommendation = rec_match.group(1).strip() if rec_match else ""
                
                if category and (insight or impact or recommendation):
                    insights.append({
                        'category': category,
                        'insight': insight,
                        'impact': impact,
                        'recommendation': recommendation
                    })
            except Exception as e:
                print(f"Error parsing section: {str(e)}")
                continue
                
        return insights

    def generate_insights(self, 
                         forecasts: Dict[str, ForecastResult],
                         budget_deltas: List[BudgetDelta],
                         strategic_goals: List[Dict]) -> List[Insight]:
        
        # Convert ForecastResult objects to dictionaries
        forecast_dicts = {
            category: {
                'subcategory': forecast.subcategory,
                'forecasted_amount': forecast.forecasted_amount,
                'confidence_interval': forecast.confidence_interval
            }
            for category, forecast in forecasts.items()
        }
        
        # Convert budget deltas to dictionary format
        budget_deltas_dict = {}
        for delta in budget_deltas:
            if isinstance(delta, dict):
                budget_deltas_dict[delta['category']] = {
                    'old_amount': delta['old_amount'],
                    'new_amount': delta['new_amount'],
                    'delta': delta['delta']
                }
            else:
                budget_deltas_dict[delta.category] = {
                    'old_amount': delta.old_amount,
                    'new_amount': delta.new_amount,
                    'delta': delta.delta
                }
        
        # Convert strategic goals to dictionary format if they're not already
        strategic_goals_dict = []
        for goal in strategic_goals:
            if isinstance(goal, dict):
                strategic_goals_dict.append({
                    'category': goal['category'],
                    'objective': goal['objective'],
                    'priority': goal['priority']
                })
            else:
                strategic_goals_dict.append({
                    'category': goal.category,
                    'objective': goal.objective,
                    'priority': goal.priority
                })
        
        # Create task
        task = Task(
            description=f"""Analyze the following budget data and generate detailed insights:
            
            Forecasts: {json.dumps(forecast_dicts, indent=2)}
            Budget Changes: {json.dumps(budget_deltas_dict, indent=2)}
            Strategic Goals: {json.dumps(strategic_goals_dict, indent=2)}
            
            For each significant change or forecast, provide a detailed analysis in the following format:
            
            Category: [Category Name]
            Insight: [Clear insight about the impact, including specific numbers and trends]
            Impact: [Detailed analysis of potential effects on educational outcomes, including both short-term and long-term implications]
            Recommendation: [Specific, actionable recommendation with clear next steps]
            
            Focus on:
            1. Changes that align with or impact strategic goals
            2. Significant budget changes (>5% or >$10,000)
            3. Categories with high strategic priority
            4. Potential risks and opportunities
            5. Impact on student outcomes and educational quality
            
            Provide quantitative analysis where possible, including:
            - Percentage changes
            - Absolute dollar amounts
            - Historical trends
            - Comparative analysis with similar categories
            
            IMPORTANT: You MUST follow this exact format for each category:
            
            Category: [Category Name]
            Insight: [Your insight here]
            Impact: [Your impact analysis here]
            Recommendation: [Your recommendation here]
            
            Do not include any other text or formatting. Each category should be analyzed separately with these exact headers.
            
            Example format:
            Category: Math Teachers
            Insight: The budget for Math Teachers has increased by 5% ($240,000 to $252,000).
            Impact: This increase will allow for additional professional development and resources.
            Recommendation: Allocate 30% of the increase to teacher training programs.
            
            Category: Smartboards
            Insight: The Smartboards budget has decreased by $10,000.
            Impact: This reduction may delay technology modernization goals.
            Recommendation: Prioritize installation in high-priority classrooms first.""",
            expected_output="A detailed analysis of budget changes and forecasts, formatted as sections for each category with insights, impacts, and recommendations.",
            agent=self.insight_agent
        )

        # Create and run crew
        crew = Crew(
            agents=[self.insight_agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()
        
        # Handle CrewOutput object
        if hasattr(result, 'raw_output'):
            output_text = result.raw_output
        else:
            output_text = str(result)
        
        # Parse the LLM output into structured insights
        parsed_insights = self._parse_llm_output(output_text)
        
        # Convert to Insight objects
        insights = []
        for insight_data in parsed_insights:
            try:
                insights.append(Insight(
                    category=insight_data['category'],
                    insight=insight_data['insight'],
                    impact=insight_data['impact'],
                    recommendation=insight_data['recommendation']
                ))
            except Exception as e:
                print(f"Error creating insight object: {str(e)}")
                continue

        return insights 