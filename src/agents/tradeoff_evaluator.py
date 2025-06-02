from crewai import Agent, Task, Crew
from typing import List, Dict
import json
from ..models.data_models import TradeOff, BudgetDelta
from langchain_openai import ChatOpenAI
import re

class TradeOffEvaluator:
    def __init__(self):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        self.tradeoff_agent = Agent(
            role='Budget Trade-off Analyst',
            goal='Evaluate trade-offs in budget changes',
            backstory="""You are a strategic budget analyst specializing in K-12 education finance.
            You excel at identifying and analyzing the complex trade-offs involved in budget decisions.
            You understand how different budget changes impact educational outcomes and can assess
            both short-term and long-term implications of budget modifications.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

    def evaluate_tradeoffs(self,
                          budget_changes: List[BudgetDelta],
                          strategic_goals: List[Dict],
                          current_budget: Dict) -> List[Dict]:
        """Evaluate trade-offs between budget changes and strategic goals."""
        try:
            # Convert budget changes to dictionary format if they're not already
            budget_changes_dict = {}
            for change in budget_changes:
                if isinstance(change, dict):
                    budget_changes_dict[change['category']] = {
                        'old_amount': change['old_amount'],
                        'new_amount': change['new_amount'],
                        'delta': change['delta']
                    }
                else:
                    budget_changes_dict[change.category] = {
                        'old_amount': change.old_amount,
                        'new_amount': change.new_amount,
                        'delta': change.delta
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
                description=f"""Analyze the following budget changes, strategic goals, and current budget to evaluate trade-offs:
                
                Budget Changes: {json.dumps(budget_changes_dict, indent=2)}
                Strategic Goals: {json.dumps(strategic_goals_dict, indent=2)}
                Current Budget: {json.dumps(current_budget, indent=2)}
                
                For each significant trade-off, provide an analysis in the following format:
                
                Category: [Category Name]
                Trade-off: [Description of the trade-off]
                Impact: [Analysis of the impact on strategic goals and educational outcomes]
                Risk Level: [High/Medium/Low]
                Mitigation: [Steps to mitigate negative impacts]
                
                Focus on:
                1. Alignment with strategic goals
                2. Impact on student outcomes
                3. Resource allocation efficiency
                4. Long-term sustainability
                5. Risk management
                
                Provide quantitative analysis where possible, including:
                - Percentage changes
                - Absolute dollar amounts
                - Impact on student outcomes
                - Cost-benefit analysis
                
                IMPORTANT: You MUST follow this exact format for each category:
                
                Category: [Category Name]
                Trade-off: [Your trade-off analysis here]
                Impact: [Your impact analysis here]
                Risk Level: [Your risk assessment here]
                Mitigation: [Your mitigation steps here]
                
                Do not include any other text or formatting. Each category should be analyzed separately with these exact headers.""",
                expected_output="A detailed analysis of trade-offs between budget changes and strategic goals, formatted as sections for each category with trade-offs, impacts, risk levels, and mitigation steps.",
                agent=self.tradeoff_agent
            )

            # Create and run crew
            crew = Crew(
                agents=[self.tradeoff_agent],
                tasks=[task],
                verbose=True
            )

            result = crew.kickoff()
            
            # Handle CrewOutput object
            if hasattr(result, 'raw_output'):
                output_text = result.raw_output
            else:
                output_text = str(result)
            
            # Parse the output into structured trade-offs
            tradeoffs = []
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
                    
                    # Extract trade-off
                    tradeoff_match = re.search(r'Trade-off:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    tradeoff = tradeoff_match.group(1).strip() if tradeoff_match else ""
                    
                    # Extract impact
                    impact_match = re.search(r'Impact:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    impact = impact_match.group(1).strip() if impact_match else ""
                    
                    # Extract risk level
                    risk_match = re.search(r'Risk Level:\s*(.+?)(?=\n|$)', section)
                    risk_level = risk_match.group(1).strip() if risk_match else ""
                    
                    # Extract mitigation
                    mitigation_match = re.search(r'Mitigation:\s*(.+?)(?=\n|$)', section, re.DOTALL)
                    mitigation = mitigation_match.group(1).strip() if mitigation_match else ""
                    
                    if category and (tradeoff or impact or risk_level or mitigation):
                        tradeoffs.append({
                            'category': category,
                            'tradeoff': tradeoff,
                            'impact': impact,
                            'risk_level': risk_level,
                            'mitigation': mitigation
                        })
                except Exception as e:
                    print(f"Error parsing section: {str(e)}")
                    continue
                    
            return tradeoffs
            
        except Exception as e:
            print(f"Error evaluating trade-offs: {str(e)}")
            return []

    def _format_budget_changes(self, budget_changes: List[BudgetDelta]) -> str:
        """Format budget changes for the task description."""
        changes = []
        for change in budget_changes:
            # Calculate percentage change
            if change.old_amount == 0:
                percentage_change = 100 if change.delta > 0 else -100 if change.delta < 0 else 0
            else:
                percentage_change = (change.delta / change.old_amount * 100)
            
            changes.append(
                f"- {change.category}: ${change.delta:,.2f} ({percentage_change:+.1f}%)"
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

    def _format_current_budget(self, current_budget: Dict) -> str:
        """Format current budget for the task description."""
        budget_items = []
        for category, amount in current_budget.items():
            budget_items.append(f"- {category}: ${amount:,.2f}")
        return "\n".join(budget_items)

    def _parse_llm_output(self, output: str) -> List[TradeOff]:
        """Parse LLM output into trade-off analyses."""
        try:
            # Extract JSON array from output
            start = output.find('[')
            end = output.rfind(']') + 1
            if start == -1 or end == 0:
                return []
                
            json_str = output[start:end]
            tradeoffs_data = json.loads(json_str)
            
            # Convert to TradeOff objects
            tradeoffs = []
            for tradeoff in tradeoffs_data:
                tradeoffs.append(TradeOff(
                    category=tradeoff['category'],
                    impact=tradeoff['impact'],
                    risk_level=tradeoff['risk_level'],
                    mitigation_strategy=tradeoff['mitigation_strategy']
                ))
            
            return tradeoffs
            
        except Exception as e:
            print(f"Error parsing trade-off analyses: {str(e)}")
            return [] 