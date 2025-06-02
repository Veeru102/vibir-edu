from fastapi import FastAPI, HTTPException
from typing import Dict, List
import os
from ..pipeline.orchestrator import PipelineOrchestrator
from ..models.data_models import Scenario, NarrativeSummary

app = FastAPI(title="VibirEdu Budget Analysis Pipeline")

# Initialize the pipeline orchestrator
orchestrator = PipelineOrchestrator(
    funding_constraints_path="data/funding_constraints.json",
    scenarios_path="data/scenario_list.json",
    snapshot_budget_path="data/snapshot_budget.csv",
    timeseries_budget_path="data/timeseries_budget.csv",
    strategic_goals_path="data/strategic_goals.json"
)

@app.get("/")
async def root():
    return {"message": "Welcome to VibirEdu Budget Analysis Pipeline"}

@app.post("/analyze-scenario/{scenario_id}")
async def analyze_scenario(scenario_id: str) -> NarrativeSummary:
    try:
        return orchestrator.process_scenario(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-all-scenarios")
async def analyze_all_scenarios() -> Dict[str, NarrativeSummary]:
    try:
        return orchestrator.process_all_scenarios()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 