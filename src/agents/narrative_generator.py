from crewai import Agent, Task, Crew
from typing import List, Dict
import json
from ..models.data_models import NarrativeSummary, Insight, OffsetRecommendation, TradeOffAnalysis
from langchain_openai import ChatOpenAI

class NarrativeGenerator:
    def __init__(self):
        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        self.narrative_agent = Agent(
            role='Budget Narrative Specialist',
            goal='Generate clear and actionable narratives from budget analyses',
            backstory="""You are a skilled communicator specializing in K-12 education finance.
            You excel at translating complex budget analyses into clear, actionable insights
            that help stakeholders understand the implications of budget decisions.
            You have a talent for highlighting key points while maintaining accuracy and nuance.
            You understand how to present information in a way that supports informed decision-making.""",
            verbose=True,
            llm=llm,
            allow_delegation=False
        )

    def generate_narrative(self,
                          scenario_id: str,
                          insights: List[Dict],
                          offsets: List[Dict],
                          tradeoffs: List[Dict],
                          strategic_goals: List[Dict]) -> NarrativeSummary:
        """Generate a narrative summary of the scenario analysis."""
        try:
            # Extract scenario ID if it's a Scenario object
            if hasattr(scenario_id, 'id'):
                scenario_id = scenario_id.id
            
            # Format insights
            formatted_insights = []
            if insights:
                for insight in insights:
                    if isinstance(insight, dict):
                        formatted_insights.append({
                            'category': insight.get('category', ''),
                            'insight': insight.get('insight', ''),
                            'impact': insight.get('impact', ''),
                            'recommendation': insight.get('recommendation', '')
                        })
                    else:
                        formatted_insights.append({
                            'category': insight.category,
                            'insight': insight.insight,
                            'impact': insight.impact,
                            'recommendation': insight.recommendation
                        })
            
            # Format offsets
            formatted_offsets = []
            if offsets:
                for offset in offsets:
                    if isinstance(offset, dict):
                        formatted_offsets.append({
                            'category': offset.get('category', ''),
                            'offset_amount': offset.get('offset_amount', ''),
                            'rationale': offset.get('rationale', ''),
                            'impact': offset.get('impact', ''),
                            'implementation': offset.get('implementation', '')
                        })
                    else:
                        formatted_offsets.append({
                            'category': offset.category,
                            'offset_amount': offset.offset_amount,
                            'rationale': offset.rationale,
                            'impact': offset.impact,
                            'implementation': offset.implementation
                        })
            
            # Format trade-offs
            formatted_tradeoffs = []
            if tradeoffs:
                for tradeoff in tradeoffs:
                    if isinstance(tradeoff, dict):
                        formatted_tradeoffs.append({
                            'category': tradeoff.get('category', ''),
                            'tradeoff': tradeoff.get('tradeoff', ''),
                            'impact': tradeoff.get('impact', ''),
                            'risk_level': tradeoff.get('risk_level', ''),
                            'mitigation': tradeoff.get('mitigation', '')
                        })
                    else:
                        formatted_tradeoffs.append({
                            'category': tradeoff.category,
                            'tradeoff': tradeoff.tradeoff,
                            'impact': tradeoff.impact,
                            'risk_level': tradeoff.risk_level,
                            'mitigation': tradeoff.mitigation
                        })
            
            # Format strategic goals
            formatted_goals = []
            if strategic_goals:
                for goal in strategic_goals:
                    if isinstance(goal, dict):
                        formatted_goals.append({
                            'category': goal.get('category', ''),
                            'objective': goal.get('objective', ''),
                            'priority': goal.get('priority', '')
                        })
                    else:
                        formatted_goals.append({
                            'category': goal.category,
                            'objective': goal.objective,
                            'priority': goal.priority
                        })
            
            # Create task with reduced verbosity
            task = Task(
                description=f"""Generate a comprehensive narrative summary for scenario {scenario_id}:

Insights:
{json.dumps(formatted_insights, indent=2)}

Offset Recommendations:
{json.dumps(formatted_offsets, indent=2)}

Trade-off Analysis:
{json.dumps(formatted_tradeoffs, indent=2)}

Strategic Goals:
{json.dumps(formatted_goals, indent=2)}

Provide a narrative that:
1. Summarizes the key findings
2. Highlights significant impacts
3. Explains the rationale for recommendations
4. Addresses potential risks and mitigations
5. Provides clear next steps

Format the output as a JSON object with fields:
- executive_summary: string (brief overview)
- key_findings: array of strings
- recommendations: array of strings
- strategic_implications: array of strings
- narrative: string (detailed analysis)

IMPORTANT: The output MUST be a valid JSON object with these exact fields.""",
                expected_output="A comprehensive narrative summary in JSON format with executive summary, key findings, recommendations, strategic implications, and detailed narrative.",
                agent=self.narrative_agent
            )

            # Create and run crew with reduced verbosity
            crew = Crew(
                agents=[self.narrative_agent],
                tasks=[task],
                verbose=False  # Set to False to reduce output verbosity
            )

            result = crew.kickoff()
            
            # Handle CrewOutput object
            if hasattr(result, 'raw_output'):
                output_text = result.raw_output
            else:
                output_text = str(result)
            
            # Try to parse the output as JSON
            try:
                narrative_data = json.loads(output_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a default narrative
                narrative_data = {
                    'executive_summary': f"Error generating narrative for scenario {scenario_id}",
                    'key_findings': ["Unable to parse narrative output"],
                    'recommendations': ["Please review the scenario analysis manually"],
                    'strategic_implications': ["Narrative generation failed"],
                    'narrative': f"Error generating narrative for scenario {scenario_id}. Please review the scenario analysis manually."
                }
            
            # Create NarrativeSummary object
            return NarrativeSummary(
                scenario_id=scenario_id,
                executive_summary=narrative_data.get('executive_summary', ''),
                key_findings=narrative_data.get('key_findings', []),
                recommendations=narrative_data.get('recommendations', []),
                strategic_implications=narrative_data.get('strategic_implications', []),
                narrative=narrative_data.get('narrative', '')
            )
            
        except Exception as e:
            print(f"Error generating narrative: {str(e)}")
            # Return a default narrative with error information
            return NarrativeSummary(
                scenario_id=scenario_id if isinstance(scenario_id, str) else getattr(scenario_id, 'id', 'unknown'),
                executive_summary=f"Error generating narrative: {str(e)}",
                key_findings=["Analysis could not be completed due to errors"],
                recommendations=["Please review the scenario manually"],
                strategic_implications=["Error in analysis pipeline"],
                narrative=f"The analysis pipeline encountered an error: {str(e)}"
            )

    def _format_insights(self, insights: List[Insight]) -> str:
        """Format insights for the task description."""
        formatted = []
        for insight in insights:
            formatted.append(
                f"- {insight.category}:\n"
                f"  Insight: {insight.insight}\n"
                f"  Impact: {insight.impact}\n"
                f"  Recommendation: {insight.recommendation}"
            )
        return "\n".join(formatted)

    def _format_offsets(self, offsets: List[Dict]) -> str:
        """Format offset recommendations for the task description."""
        formatted = []
        for offset in offsets:
            formatted.append(
                f"- {offset['category']}:\n"
                f"  Amount: ${offset['amount']:,.2f}\n"
                f"  Reason: {offset['reason']}\n"
                f"  Priority: {offset['priority']}"
            )
        return "\n".join(formatted)

    def _format_tradeoffs(self, tradeoffs: List[TradeOffAnalysis]) -> str:
        """Format trade-off analyses for the task description."""
        formatted = []
        for tradeoff in tradeoffs:
            formatted.append(
                f"- {tradeoff.category}:\n"
                f"  Impact: {tradeoff.impact}\n"
                f"  Risk Level: {tradeoff.risk_level}\n"
                f"  Mitigation: {tradeoff.mitigation_strategies}"
            )
        return "\n".join(formatted) 