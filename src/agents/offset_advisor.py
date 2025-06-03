from crewai import Agent, Task, Crew
from typing import List, Dict, Tuple
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
            # Calculate net delta (total increase in spending)
            net_delta = sum(delta.delta if isinstance(delta, BudgetDelta) else delta['delta'] 
                          for delta in budget_deltas)
            
            if net_delta <= 0:
                return []  # No need for offsets if there's no net increase
                
            # Get current budget snapshot
            current_budget = self._get_current_budget()
            
            # Get candidate categories for offsets
            candidate_categories = self._get_candidate_categories(
                budget_deltas,
                current_budget,
                strategic_goals
            )
            
            # Select offset sources
            offset_sources = self._select_offset_sources(
                net_delta,
                candidate_categories,
                current_budget,
                strategic_goals
            )
            
            # Generate detailed recommendations using LLM
            recommendations = self._generate_detailed_recommendations(
                offset_sources,
                budget_deltas,
                strategic_goals
            )
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating offset recommendations: {str(e)}")
            return []

    def _get_current_budget(self) -> Dict[str, float]:
        """Get current budget snapshot."""
        # This should be implemented to get the current budget state
        # For now, return an empty dict
        return {}

    def _get_candidate_categories(self,
                                budget_deltas: List[BudgetDelta],
                                current_budget: Dict[str, float],
                                strategic_goals: List[Dict]) -> List[Dict]:
        """Get list of candidate categories for offsets."""
        candidates = []
        
        # Get categories that were increased
        increased_categories = {
            delta.category if isinstance(delta, BudgetDelta) else delta['category']
            for delta in budget_deltas
            if (isinstance(delta, BudgetDelta) and delta.delta > 0) or
               (isinstance(delta, dict) and delta['delta'] > 0)
        }
        
        # Get categories that are locked
        locked_categories = set()
        if hasattr(self.funding_constraints, 'locked_categories'):
            locked_categories = set(self.funding_constraints.locked_categories)
        elif isinstance(self.funding_constraints, dict):
            locked_categories = {
                category for category, constraint in self.funding_constraints.items()
                if constraint.get('locked', False)
            }
        
        # Create candidate list
        for category, amount in current_budget.items():
            # Skip if category was increased or is locked
            if category in increased_categories or category in locked_categories:
                continue
                
            # Get strategic goal priority
            goal = next(
                (g for g in strategic_goals if g['category'] == category),
                {'priority': 'medium'}  # Default priority if not found
            )
            
            candidates.append({
                'category': category,
                'current_amount': amount,
                'priority': goal['priority'],
                'objective': goal.get('objective', '')
            })
        
        return candidates

    def _select_offset_sources(self,
                             net_delta: float,
                             candidates: List[Dict],
                             current_budget: Dict[str, float],
                             strategic_goals: List[Dict]) -> List[Dict]:
        """Select categories to offset the net delta."""
        # Sort candidates by priority (low first) and current amount (high to low)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                {'low': 0, 'medium': 1, 'high': 2}[x['priority']],
                -x['current_amount']  # Negative for descending order
            )
        )
        
        offset_sources = []
        remaining_delta = net_delta
        
        for candidate in sorted_candidates:
            if remaining_delta <= 0:
                break
                
            # Calculate maximum possible offset (up to 20% of current amount)
            max_offset = min(
                candidate['current_amount'] * 0.2,  # Max 20% of current amount
                remaining_delta  # Don't exceed remaining delta
            )
            
            if max_offset > 0:
                offset_sources.append({
                    'category': candidate['category'],
                    'offset_amount': max_offset,
                    'priority': candidate['priority'],
                    'objective': candidate['objective']
                })
                remaining_delta -= max_offset
        
        return offset_sources

    def _generate_detailed_recommendations(self,
                                        offset_sources: List[Dict],
                                        budget_deltas: List[BudgetDelta],
                                        strategic_goals: List[Dict]) -> List[Dict]:
        """Generate detailed offset recommendations using LLM."""
        # Create task for LLM
        task = Task(
            description=f"""Generate detailed offset recommendations for the following sources:
            
            Offset Sources: {json.dumps(offset_sources, indent=2)}
            Budget Changes: {json.dumps([d.dict() if isinstance(d, BudgetDelta) else d for d in budget_deltas], indent=2)}
            Strategic Goals: {json.dumps(strategic_goals, indent=2)}
            
            For each offset source, provide a detailed recommendation in the following format:
            
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