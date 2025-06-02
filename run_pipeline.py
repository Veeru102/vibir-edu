from src.pipeline.orchestrator import PipelineOrchestrator
import sys

def main():
    try:
        # Initialize the pipeline orchestrator
        orchestrator = PipelineOrchestrator(
            scenarios_path="data/scenario_list.json",
            funding_constraints_path="data/funding_constraints.json",
            strategic_goals_path="data/strategic_goals.json",
            snapshot_budget_path="data/snapshot_budget.csv",
            timeseries_budget_path="data/timeseries_budget.csv"
        )
        
        # Process all scenarios
        results = orchestrator.process_all_scenarios()
        
        # Print results in a user-friendly format
        orchestrator.print_results(results)
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 