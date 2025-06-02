from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Union
from datetime import datetime

class FundingConstraint(BaseModel):
    categories: List[str]
    locked_categories: List[str]
    note: str

class Scenario(BaseModel):
    id: str
    target_category: str
    source_fund: str
    is_mandated: bool
    is_reversible: bool
    reason_for_change: str
    percentage: Optional[float] = None
    fixed_delta: Optional[float] = None
    defer_months: Optional[int] = None

    @validator('percentage', 'fixed_delta', 'defer_months')
    def validate_change_type(cls, v, values):
        """Ensure only one type of change is specified."""
        if v is not None:
            # Check if any other change type is set
            if 'percentage' in values and values['percentage'] is not None and values['percentage'] != v:
                raise ValueError("Only one of percentage, fixed_delta, or defer_months can be set")
            if 'fixed_delta' in values and values['fixed_delta'] is not None and values['fixed_delta'] != v:
                raise ValueError("Only one of percentage, fixed_delta, or defer_months can be set")
            if 'defer_months' in values and values['defer_months'] is not None and values['defer_months'] != v:
                raise ValueError("Only one of percentage, fixed_delta, or defer_months can be set")
        return v

    @property
    def type(self) -> str:
        """Get the type of change."""
        if self.percentage is not None:
            return 'percentage'
        elif self.fixed_delta is not None:
            return 'fixed'
        elif self.defer_months is not None:
            return 'deferral'
        raise ValueError("No change type specified")

    @property
    def value(self) -> Union[float, int]:
        """Get the value of the change."""
        if self.percentage is not None:
            return self.percentage
        elif self.fixed_delta is not None:
            return self.fixed_delta
        elif self.defer_months is not None:
            return self.defer_months
        raise ValueError("No change value specified")

class ForecastResult(BaseModel):
    subcategory: str
    forecasted_amount: float
    confidence_interval: Dict[str, float]

class Insight(BaseModel):
    category: str
    insight: str
    impact: str
    recommendation: str

class TradeOff(BaseModel):
    category: str
    impact: str
    risk_level: str
    mitigation_strategy: str

class StrategicGoal(BaseModel):
    """Strategic goal for budget planning."""
    category: str
    objective: str
    priority: str
    goal_type: str
    horizon: str

    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['high', 'medium', 'low']
        if v.lower() not in valid_priorities:
            raise ValueError(f'Priority must be one of {valid_priorities}')
        return v.lower()

    @validator('goal_type')
    def validate_goal_type(cls, v):
        valid_types = ['performance', 'equity', 'access', 'efficiency']
        if v.lower() not in valid_types:
            raise ValueError(f'Goal type must be one of {valid_types}')
        return v.lower()

    @validator('horizon')
    def validate_horizon(cls, v):
        valid_horizons = ['short-term', 'medium-term', 'long-term']
        if v.lower() not in valid_horizons:
            raise ValueError(f'Horizon must be one of {valid_horizons}')
        return v.lower()

    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'objective': self.objective,
            'priority': self.priority,
            'goal_type': self.goal_type,
            'horizon': self.horizon
        }

class NarrativeSummary(BaseModel):
    scenario_id: str
    executive_summary: str
    key_findings: List[str]
    recommendations: List[str]
    strategic_implications: List[str]
    narrative: str

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]

class BudgetEntry(BaseModel):
    subcategory: str
    amount: float
    year: int
    amount_type: str

    @validator('amount_type')
    def validate_amount_type(cls, v):
        valid_types = ['Annual', 'Monthly', 'Quarterly']
        if v not in valid_types:
            raise ValueError(f'Amount type must be one of {valid_types}')
        return v

class BudgetSnapshot(BaseModel):
    subcategory: List[BudgetEntry]

    @validator('subcategory')
    def validate_subcategory(cls, v):
        if not v:
            raise ValueError('Budget snapshot must contain at least one entry')
        return v

class BudgetDelta(BaseModel):
    category: str
    old_amount: float
    new_amount: float
    delta: float

    @property
    def percentage_change(self) -> float:
        """Calculate the percentage change from old_amount to new_amount."""
        if self.old_amount == 0:
            return 100 if self.delta > 0 else -100 if self.delta < 0 else 0
        return (self.delta / self.old_amount * 100)

class ScenarioValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class OffsetRecommendation(BaseModel):
    category: str
    amount: float
    reason: str
    priority: str

class TradeOffAnalysis(BaseModel):
    category: str
    impact: str
    risk_level: str
    mitigation_strategies: str
    long_term_implications: str

class TimeSeriesEntry(BaseModel):
    start_date: datetime
    subcategory: str
    amount: float

    @validator('start_date')
    def validate_date(cls, v):
        if not isinstance(v, datetime):
            try:
                return datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v 