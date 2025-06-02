from crewai import Agent, Task, Crew
from typing import List, Dict
import json
import re
from ..models.data_models import OffsetRecommendation, FundingConstraint, BudgetDelta
from langchain_openai import ChatOpenAI

class OffsetAdvisor:
    def __init__(self, funding_constraints: Dict):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        self.offset_agent = Agent(
            role='Budget Offset Advisor',
            goal='Recommend appropriate budget offsets to balance changes',
            backstory="""You are a strategic budget advisor with deep experience in K-12 education finance.
            You excel at identifying potential areas for budget adjustments while maintaining educational quality.
            You understand the complex relationships between different budget categories and their impact on student outcomes.
            You can identify both short-term and long-term opportunities for budget optimization.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        self.funding_constraints = funding_constraints

    def get_offset_recommendations(self, 
                                 budget_deltas: List[BudgetDelta],
                                 strategic_goals: List[Dict]) -> List[Dict]:
        """Get offset recommendations for budget changes."""
        try:
            # Convert budget deltas to dictionary format if they're not already
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
                    strategic_goals_dict.append(goal)
                else:
                    strategic_goals_dict.append({
                        'category': goal.category,
                        'objective': goal.objective,
                        'priority': goal.priority
                    })
            
            # Create task
            task = Task(
                description=f"""Analyze the following budget changes and strategic goals to provide offset recommendations:
                
                Budget Changes: {json.dumps(budget_deltas_dict, indent=2)}
                Strategic Goals: {json.dumps(strategic_goals_dict, indent=2)}
                
                For each significant budget change, provide offset recommendations in the following format:
                
                Category: [Category Name]
                Offset Amount: [Amount to offset]
                Rationale: [Clear explanation of why this offset is recommended]
                Impact: [Analysis of the impact on strategic goals and educational outcomes]
                Implementation: [Specific steps to implement the offset]
                
                Focus on:
                1. Aligning offsets with strategic goals
                2. Minimizing negative impact on educational outcomes
                3. Ensuring sustainable budget adjustments
                4. Maintaining service quality
                5. Supporting student success
                
                Provide quantitative analysis where possible, including:
                - Percentage changes
                - Absolute dollar amounts
                - Impact on student outcomes
                - Cost-benefit analysis
                
                IMPORTANT: You MUST follow this exact format for each category:
                
                Category: [Category Name]
                Offset Amount: [Your offset amount here]
                Rationale: [Your rationale here]
                Impact: [Your impact analysis here]
                Implementation: [Your implementation steps here]
                
                Do not include any other text or formatting. Each category should be analyzed separately with these exact headers.""",
                expected_output="A detailed analysis of offset recommendations for budget changes, formatted as sections for each category with offset amounts, rationales, impacts, and implementation steps.",
                agent=self.offset_agent
            )

            # Create and run crew
            crew = Crew(
                agents=[self.offset_agent],
                tasks=[task],
                verbose=True
            )

            result = crew.kickoff()
            
            # Handle CrewOutput object
            if hasattr(result, 'raw_output'):
                output_text = result.raw_output
            else:
                output_text = str(result)
            
            # Parse the output into structured recommendations
            recommendations = []
            sections = output_text.split('\n\n')
            
            for section in sections:
                if not section.strip():
                    continue
                    
                try:
                    # Extract category
                    category_match = re.search(r'Category:\s*(.+?)(?=\n|$)', section)
                    if not category_match:
                        continue
                    category = category_match.group(1).strip()
                    
                    # Extract offset amount
                    amount_match = re.search(r'Offset Amount:\s*(.+?)(?=\n|$)', section)
                    offset_amount = amount_match.group(1).strip() if amount_match else ""
                    
                    # Extract rationale
                    rationale_match = re.search(r'Rationale:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    rationale = rationale_match.group(1).strip() if rationale_match else ""
                    
                    # Extract impact
                    impact_match = re.search(r'Impact:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    impact = impact_match.group(1).strip() if impact_match else ""
                    
                    # Extract implementation
                    impl_match = re.search(r'Implementation:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    implementation = impl_match.group(1).strip() if impl_match else ""
                    
                    if category and (offset_amount or rationale or impact or implementation):
                        recommendations.append({
                            'category': category,
                            'offset_amount': offset_amount,
                            'rationale': rationale,
                            'impact': impact,
                            'implementation': implementation
                        })
                except Exception as e:
                    print(f"Error parsing section: {str(e)}")
                    continue
                    
            return recommendations
            
        except Exception as e:
            print(f"Error generating offset recommendations: {str(e)}")
            return []

    def _format_budget_changes(self, budget_deltas: List[Dict]) -> str:
        """Format budget changes for the task description."""
        changes = []
        for delta in budget_deltas:
            # Calculate percentage change
            if delta['old_amount'] == 0:
                percentage_change = 100 if delta['delta'] > 0 else -100 if delta['delta'] < 0 else 0
            else:
                percentage_change = (delta['delta'] / delta['old_amount'] * 100)
            
            changes.append(
                f"- {delta['category']}: ${delta['delta']:,.2f} ({percentage_change:+.1f}%)"
            )
        return "\n".join(changes)

    def _format_strategic_goals(self, strategic_goals: List[Dict]) -> str:
        """Format strategic goals for the task description."""
        goals = []
        for goal in strategic_goals:
            goals.append(
                f"- {goal['category']}: {goal['objective']} (Priority: {goal['priority']})"
            )
        return "\n".join(goals)

    def _format_funding_constraints(self) -> str:
        """Format funding constraints for the task description."""
        constraints = []
        for category, constraint in self.funding_constraints.items():
            status = "Locked" if constraint['locked'] else "Flexible"
            constraints.append(f"- {category}: {status} - {constraint['note']}")
        return "\n".join(constraints)

    def _parse_llm_output(self, output: str) -> List[OffsetRecommendation]:
        """Parse LLM output into offset recommendations."""
        try:
            # Extract JSON array from output
            start = output.find('[')
            end = output.rfind(']') + 1
            if start == -1 or end == 0:
                return []
                
            json_str = output[start:end]
            recommendations_data = json.loads(json_str)
            
            # Convert to OffsetRecommendation objects
            recommendations = []
            for rec in recommendations_data:
                recommendations.append(OffsetRecommendation(
                    category=rec['category'],
                    amount=float(rec['amount']),
                    reason=rec['reason'],
                    priority=rec['priority']
                ))
            
            return recommendations
            
        except Exception as e:
            print(f"Error parsing offset recommendations: {str(e)}")
            return [] 