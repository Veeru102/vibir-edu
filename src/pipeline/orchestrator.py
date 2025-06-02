from typing import List, Dict
import json
from ..pipeline.scenario_loader import ScenarioLoader
from ..pipeline.budget_applier import BudgetScenarioApplier
from ..pipeline.cost_forecaster import CostForecaster
from ..agents.insight_generator import InsightGenerator
from ..agents.offset_advisor import OffsetAdvisor
from ..agents.tradeoff_evaluator import TradeOffEvaluator
from ..agents.narrative_generator import NarrativeGenerator
from ..models.data_models import Scenario, NarrativeSummary, ForecastResult, StrategicGoal, BudgetDelta

class PipelineOrchestrator:
    def __init__(self,
                 funding_constraints_path: str,
                 scenarios_path: str,
                 snapshot_budget_path: str,
                 timeseries_budget_path: str,
                 strategic_goals_path: str):
        
        # Initialize components
        self.scenario_loader = ScenarioLoader(
            funding_constraints_path=funding_constraints_path,
            scenarios_path=scenarios_path
        )
        self.budget_applier = BudgetScenarioApplier(snapshot_budget_path)
        self.cost_forecaster = CostForecaster(timeseries_budget_path)
        self.insight_generator = InsightGenerator()
        self.offset_advisor = OffsetAdvisor(self.scenario_loader.funding_constraints)
        self.tradeoff_evaluator = TradeOffEvaluator()
        self.narrative_generator = NarrativeGenerator()
        
        # Store paths for later use
        self.scenarios_path = scenarios_path
        self.snapshot_budget_path = snapshot_budget_path
        self.timeseries_budget_path = timeseries_budget_path
        self.strategic_goals_path = strategic_goals_path
        
        # Load strategic goals and convert to objects
        with open(strategic_goals_path, 'r') as f:
            goals_data = json.load(f)['goals']
            self.strategic_goals = [StrategicGoal(**goal) for goal in goals_data]

    def process_scenario(self, scenario_id: str) -> NarrativeSummary:
        """Process a single scenario and generate a narrative summary."""
        try:
            print(f"\nProcessing scenario: {scenario_id}")
            print("=" * 80)
            
            # Load and validate scenario
            print("\n1. Loading and validating scenario...")
            scenario = self.scenario_loader.load_scenario(scenario_id)
            if not scenario:
                raise ValueError(f"Failed to load scenario {scenario_id}")
            print(f"Loaded scenario: {scenario.dict()}")
            
            if not self.scenario_loader.validate_scenario(scenario):
                raise ValueError(f"Scenario {scenario_id} is invalid")
            print("Scenario validation passed")
            
            # Take budget snapshot
            print("\n2. Taking budget snapshot...")
            self.budget_snapshot = self.budget_applier.take_snapshot()
            print(f"Budget snapshot: {self.budget_snapshot.dict()}")
            
            # Apply budget changes
            print("\n3. Applying budget changes...")
            budget_deltas = self.budget_applier.apply_changes(scenario)
            print(f"Budget deltas: {[delta.dict() for delta in budget_deltas]}")
            
            # Generate forecast
            print("\n4. Generating forecast...")
            forecast_dicts = self.cost_forecaster.generate_forecasts(budget_deltas)
            forecast_results = {
                category: ForecastResult(**forecast)
                for category, forecast in forecast_dicts.items()
            }
            print(f"Forecast results: {[result.dict() for result in forecast_results.values()]}")
            
            # Generate insights
            print("\n5. Generating insights...")
            insights = self.insight_generator.generate_insights(
                forecast_results,
                budget_deltas,
                self.strategic_goals
            )
            print(f"Generated insights: {[insight.dict() for insight in insights]}")
            
            # Get offset recommendations
            print("\n6. Getting offset recommendations...")
            offset_recommendations = self.offset_advisor.get_offset_recommendations(
                budget_deltas,
                self.strategic_goals
            )
            print(f"Offset recommendations: {offset_recommendations}")
            
            # Evaluate trade-offs
            print("\n7. Evaluating trade-offs...")
            trade_offs = self.tradeoff_evaluator.evaluate_tradeoffs(
                budget_deltas,
                self.strategic_goals,
                self.budget_applier.get_current_budget()
            )
            print(f"Trade-off analysis: {trade_offs}")
            
            # Generate narrative
            print("\n8. Generating narrative...")
            narrative = self.narrative_generator.generate_narrative(
                scenario,
                insights,
                offset_recommendations,
                trade_offs,
                self.strategic_goals
            )
            print(f"Generated narrative: {narrative.dict()}")
            
            # Reset budget to snapshot
            print("\n9. Resetting budget to snapshot...")
            self.budget_applier.reset_to_snapshot(self.budget_snapshot)
            print("Budget reset complete")
            
            return narrative
            
        except Exception as e:
            print(f"\nError processing scenario {scenario_id}: {str(e)}")
            print("Resetting budget to snapshot state...")
            if hasattr(self, 'budget_snapshot'):
                self.budget_applier.reset_to_snapshot(self.budget_snapshot)
            
            # Return a default narrative with error information
            return NarrativeSummary(
                scenario_id=scenario_id,
                executive_summary=f"Error processing scenario: {str(e)}",
                key_findings=["Analysis could not be completed due to errors"],
                recommendations=["Please review the scenario manually"],
                strategic_implications=["Error in analysis pipeline"],
                narrative=f"The analysis pipeline encountered an error: {str(e)}"
            )

    def process_all_scenarios(self) -> Dict[str, NarrativeSummary]:
        """Process all scenarios and return a dictionary of narrative summaries."""
        results = {}
        scenario_ids = self.scenario_loader.get_scenario_ids()
        
        print(f"\nFound {len(scenario_ids)} scenarios to process")
        print("=" * 80)
        
        for scenario_id in scenario_ids:
            try:
                print(f"\nProcessing scenario {scenario_id}...")
                narrative = self.process_scenario(scenario_id)
                results[scenario_id] = narrative
                
                # Print detailed analysis
                print("\n" + "=" * 80)
                print(f"Scenario {scenario_id} Analysis")
                print("=" * 80)
                print("\nExecutive Summary:")
                print("-" * 40)
                print(narrative.executive_summary)
                
                print("\nKey Findings:")
                print("-" * 40)
                for finding in narrative.key_findings:
                    print(f"• {finding}")
                
                print("\nRecommendations:")
                print("-" * 40)
                for rec in narrative.recommendations:
                    print(f"• {rec}")
                
                print("\nStrategic Implications:")
                print("-" * 40)
                for impl in narrative.strategic_implications:
                    print(f"• {impl}")
                
                print("\nDetailed Analysis:")
                print("-" * 40)
                print(narrative.narrative)
                print("\n" + "=" * 80)
                
            except Exception as e:
                print(f"Error processing scenario {scenario_id}: {str(e)}")
                results[scenario_id] = NarrativeSummary(
                    scenario_id=scenario_id,
                    executive_summary=f"Error processing scenario: {str(e)}",
                    key_findings=["Analysis could not be completed due to errors"],
                    recommendations=["Please review the scenario manually"],
                    strategic_implications=["Error in analysis pipeline"],
                    narrative=f"The analysis pipeline encountered an error: {str(e)}"
                )
        
        return results

    def print_results(self, results: Dict[str, NarrativeSummary]):
        """Print results in a user-friendly format."""
        for scenario_id, result in results.items():
            print("\n" + "="*80)
            print(f"Scenario {scenario_id} Analysis")
            print("="*80)
            
            print("\nExecutive Summary:")
            print("-"*40)
            print(result.executive_summary)
            
            print("\nKey Findings:")
            print("-"*40)
            for finding in result.key_findings:
                print(f"• {finding}")
            
            print("\nRecommendations:")
            print("-"*40)
            for rec in result.recommendations:
                print(f"• {rec}")
            
            print("\nStrategic Implications:")
            print("-"*40)
            for impl in result.strategic_implications:
                print(f"• {impl}")
            
            print("\nDetailed Analysis:")
            print("-"*40)
            print(result.narrative)
            print("\n" + "="*80 + "\n") 