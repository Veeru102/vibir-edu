import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, List
from ..models.data_models import ForecastResult, TimeSeriesEntry, BudgetDelta

class CostForecaster:
    def __init__(self, timeseries_budget_path: str):
        """Initialize with path to timeseries budget data."""
        self.timeseries_budget_path = timeseries_budget_path
        self.timeseries_data = self._load_timeseries_data()
        self.models = {}

    def _load_timeseries_data(self) -> pd.DataFrame:
        """Load timeseries budget data from CSV."""
        try:
            return pd.read_csv(self.timeseries_budget_path)
        except Exception as e:
            print(f"Error loading timeseries data: {str(e)}")
            return pd.DataFrame()

    def prepare_data(self, category: str) -> pd.DataFrame:
        category_data = self.timeseries_data[self.timeseries_data['Subcategory'] == category].copy()
        category_data = category_data.rename(columns={'StartDate': 'ds', 'Amount': 'y'})
        return category_data

    def train_model(self, category: str) -> None:
        if category not in self.models:
            df = self.prepare_data(category)
            if not df.empty:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                model.fit(df)
                self.models[category] = model

    def forecast(self, category: str, periods: int = 12) -> Dict:
        if category not in self.models:
            self.train_model(category)

        if category not in self.models:  # If still not in models, no data available
            return {
                'subcategory': category,
                'forecasted_amount': 0.0,
                'confidence_interval': {'lower': 0.0, 'upper': 0.0}
            }

        model = self.models[category]
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)

        # Get the last forecasted value and its confidence interval
        last_forecast = forecast.iloc[-1]
        
        return {
            'subcategory': category,
            'forecasted_amount': float(last_forecast['yhat']),
            'confidence_interval': {
                'lower': float(last_forecast['yhat_lower']),
                'upper': float(last_forecast['yhat_upper'])
            }
        }

    def forecast_all_categories(self, periods: int = 12) -> Dict[str, Dict]:
        categories = self.timeseries_data['Subcategory'].unique()
        forecasts = {}
        
        for category in categories:
            forecasts[category] = self.forecast(category, periods)
        
        return forecasts

    def apply_deferral(self, category: str, defer_months: int) -> Dict:
        """Apply a deferral to a category's forecast"""
        if category not in self.models:
            self.train_model(category)

        if category not in self.models:  # If still not in models, no data available
            return {
                'subcategory': category,
                'forecasted_amount': 0.0,
                'confidence_interval': {'lower': 0.0, 'upper': 0.0}
            }

        model = self.models[category]
        future = model.make_future_dataframe(periods=defer_months, freq='M')
        forecast = model.predict(future)

        # Get the forecasted value after the deferral period
        deferred_forecast = forecast.iloc[-1]
        
        return {
            'subcategory': category,
            'forecasted_amount': float(deferred_forecast['yhat']),
            'confidence_interval': {
                'lower': float(deferred_forecast['yhat_lower']),
                'upper': float(deferred_forecast['yhat_upper'])
            }
        }

    def generate_forecasts(self, budget_deltas: List[BudgetDelta]) -> Dict[str, Dict]:
        """
        Generate cost forecasts based on budget changes.
        
        Args:
            budget_deltas: List of budget changes (BudgetDelta objects or dictionaries)
            
        Returns:
            Dictionary of forecast results by category
        """
        # Convert dictionaries to BudgetDelta objects if needed
        processed_deltas = []
        for delta in budget_deltas:
            if isinstance(delta, dict):
                processed_deltas.append(BudgetDelta(
                    category=delta['category'],
                    old_amount=delta['old_amount'],
                    new_amount=delta['new_amount'],
                    delta=delta['delta']
                ))
            else:
                processed_deltas.append(delta)

        if not self.timeseries_data.empty:
            return self._generate_timeseries_forecasts(processed_deltas)
        else:
            return self._generate_simple_forecasts(processed_deltas)

    def _generate_timeseries_forecasts(self, budget_deltas: List[BudgetDelta]) -> Dict[str, Dict]:
        """Generate forecasts using timeseries data."""
        forecasts = {}
        
        for delta in budget_deltas:
            # Get historical data for the category
            category_data = self.timeseries_data[
                self.timeseries_data['Subcategory'] == delta.category
            ]
            
            if category_data.empty:
                # Fall back to simple forecast if no historical data
                forecasts[delta.category] = self._create_simple_forecast(delta)
                continue
                
            # Calculate forecast using historical data
            mean_amount = category_data['Amount'].mean()
            std_amount = category_data['Amount'].std()
            
            # Apply the delta to the mean
            forecasted_amount = mean_amount + delta.delta
            
            # Create forecast result
            forecasts[delta.category] = {
                'subcategory': delta.category,
                'forecasted_amount': float(forecasted_amount),
                'confidence_interval': {
                    'lower': float(forecasted_amount - 2 * std_amount),
                    'upper': float(forecasted_amount + 2 * std_amount)
                }
            }
            
        return forecasts

    def _generate_simple_forecasts(self, budget_deltas: List[BudgetDelta]) -> Dict[str, Dict]:
        """Generate simple forecasts without timeseries data."""
        return {
            delta.category: self._create_simple_forecast(delta)
            for delta in budget_deltas
        }

    def _create_simple_forecast(self, delta: BudgetDelta) -> Dict:
        """Create a simple forecast for a budget delta."""
        return {
            'subcategory': delta.category,
            'forecasted_amount': float(delta.new_amount),
            'confidence_interval': {
                'lower': float(delta.new_amount * 0.9),  # 10% lower
                'upper': float(delta.new_amount * 1.1)   # 10% higher
            }
        } 