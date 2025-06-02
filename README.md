# VibirEdu Budget Analysis Pipeline

A sophisticated budget analysis pipeline for K-12 education institutions, powered by AI agents and machine learning.

## Overview

VibirEdu is a fiscal co-pilot for the K-12 Education Market, providing intelligent budget analysis and recommendations. The system uses a combination of rule-based logic and AI agents to analyze budget scenarios, generate insights, and provide actionable recommendations.

## Features

1. **Scenario Loader & Validator**
   - Validates budget scenarios against funding constraints
   - Uses fuzzy matching for category validation
   - Ensures compliance with funding rules

2. **Budget Scenario Applier**
   - Applies percentage/fixed/delay deltas to budget rows
   - Handles complex budget modifications
   - Maintains budget integrity

3. **Cost Forecaster**
   - Uses Prophet for time-series forecasting
   - Provides confidence intervals for predictions
   - Handles seasonal patterns and trends

4. **AI-Powered Analysis**
   - Insight Generation
   - Offset Recommendations
   - Strategic Trade-off Analysis
   - Narrative Summary Generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vibir-edu.git
cd vibir-edu
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the API server:
```bash
python -m src.api.main
```

2. The API will be available at `http://localhost:8000`

3. Available endpoints:
   - `GET /`: Welcome message
   - `POST /analyze-scenario/{scenario_id}`: Analyze a specific scenario
   - `POST /analyze-all-scenarios`: Analyze all scenarios

## Project Structure

```
vibir-edu/
├── src/
│   ├── agents/
│   │   ├── insight_generator.py
│   │   ├── offset_advisor.py
│   │   ├── tradeoff_evaluator.py
│   │   └── narrative_generator.py
│   ├── models/
│   │   └── data_models.py
│   ├── pipeline/
│   │   ├── scenario_loader.py
│   │   ├── budget_applier.py
│   │   ├── cost_forecaster.py
│   │   └── orchestrator.py
│   └── api/
│       └── main.py
├── data/
│   ├── funding_constraints.json
│   ├── scenario_list.json
│   ├── snapshot_budget.csv
│   ├── timeseries_budget.csv
│   └── strategic_goals.json
├── requirements.txt
└── README.md
```

## Data Format

### Funding Constraints
```json
{
  "title_i_grant": {
    "categories": ["Special Education Staff", "Instructional Aides"],
    "locked": true,
    "note": "Restricted federal funds"
  }
}
```

### Scenarios
```json
[
  {
    "id": "raise_math_teachers",
    "target_category": "Math Teachers",
    "percentage": 0.05
  }
]
```

### Strategic Goals
```json
{
  "goals": [
    {
      "category": "Math Teachers",
      "objective": "Improve Algebra I pass rates",
      "priority": "high"
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.