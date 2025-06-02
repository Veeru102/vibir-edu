import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path
from ..models.data_models import Scenario, BudgetSnapshot, BudgetDelta, BudgetEntry

class BudgetScenarioApplier:
    def __init__(self, snapshot_budget_path: str):
        """Initialize with path to snapshot budget CSV."""
        self.snapshot_budget_path = snapshot_budget_path
        self.current_budget = self._load_budget()
        self.snapshot = None

    def _load_budget(self) -> pd.DataFrame:
        """Load budget from CSV file."""
        try:
            df = pd.read_csv(self.snapshot_budget_path)
            # Rename columns to match expected format
            column_mapping = {
                'Subcategory': 'subcategory',
                'Amount': 'amount',
                'Year': 'year',
                'AmountType': 'amount_type'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_columns = ['subcategory', 'amount', 'year', 'amount_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded budget with {len(df)} categories")
            return df
        except Exception as e:
            print(f"Error loading budget: {str(e)}")
            return pd.DataFrame()

    def take_snapshot(self) -> BudgetSnapshot:
        """Take a snapshot of the current budget state."""
        try:
            if self.current_budget.empty:
                raise ValueError("No budget data available")
            
            snapshot_data = []
            for _, row in self.current_budget.iterrows():
                try:
                    entry = BudgetEntry(
                        subcategory=str(row['subcategory']),
                        amount=float(row['amount']),
                        year=int(row['year']),
                        amount_type=str(row['amount_type'])
                    )
                    snapshot_data.append(entry)
                except Exception as e:
                    print(f"Error creating budget entry: {str(e)}")
                    print(f"Row data: {row.to_dict()}")
                    raise
            
            if not snapshot_data:
                raise ValueError("No valid budget entries found")
            
            self.snapshot = BudgetSnapshot(subcategory=snapshot_data)
            return self.snapshot
        except Exception as e:
            print(f"Error taking budget snapshot: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller

    def reset_to_snapshot(self, snapshot: BudgetSnapshot):
        """Reset budget to a previous snapshot state."""
        try:
            if not snapshot or not snapshot.subcategory:
                print("No valid snapshot to reset to")
                return
            
            # Convert snapshot data back to DataFrame
            snapshot_data = []
            for entry in snapshot.subcategory:
                snapshot_data.append({
                    'subcategory': entry.subcategory,
                    'amount': entry.amount,
                    'year': entry.year,
                    'amount_type': entry.amount_type
                })
            self.current_budget = pd.DataFrame(snapshot_data)
            print("Budget reset to snapshot state")
        except Exception as e:
            print(f"Error resetting budget: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller

    def apply_changes(self, scenario: Scenario) -> List[BudgetDelta]:
        """Apply budget changes based on scenario."""
        try:
            if self.current_budget.empty:
                raise ValueError("No budget data loaded")

            # Find the target category in the budget
            target_rows = self.current_budget[self.current_budget['subcategory'] == scenario.target_category]
            if target_rows.empty:
                raise ValueError(f"Target category '{scenario.target_category}' not found in budget")

            deltas = []
            for _, row in target_rows.iterrows():
                old_amount = float(row['amount'])
                
                # Calculate new amount based on scenario type
                if scenario.type == 'percentage':
                    new_amount = old_amount * (1 + scenario.value)
                elif scenario.type == 'fixed':
                    new_amount = old_amount + scenario.value
                elif scenario.type == 'deferral':
                    # For deferral, we'll reduce the current amount and add it back later
                    new_amount = old_amount * (1 - (scenario.value / 12))  # Monthly reduction
                else:
                    raise ValueError(f"Invalid scenario type: {scenario.type}")

                # Update the budget
                self.current_budget.loc[self.current_budget['subcategory'] == scenario.target_category, 'amount'] = new_amount

                # Create delta record
                delta = BudgetDelta(
                    category=scenario.target_category,
                    old_amount=old_amount,
                    new_amount=new_amount,
                    delta=new_amount - old_amount
                )
                deltas.append(delta)

            return deltas

        except Exception as e:
            print(f"Error applying budget changes: {str(e)}")
            return []

    def get_current_budget(self) -> pd.DataFrame:
        """Get the current state of the budget."""
        return self.current_budget.copy()

    def apply_scenario(self, scenario: Scenario) -> List[BudgetDelta]:
        """Apply a single scenario and return the budget changes."""
        if not self.current_budget.empty:
            # Validate scenario
            if scenario.target_category not in self.current_budget['subcategory'].values:
                print(f"Target category '{scenario.target_category}' not found in budget")
                print(f"Available categories: {self.current_budget['subcategory'].tolist()}")
                return []

            # Calculate new amount based on scenario type
            target_rows = self.current_budget[self.current_budget['subcategory'] == scenario.target_category]
            if target_rows.empty:
                print(f"Target category '{scenario.target_category}' not found in budget")
                return []
            
            old_amount = float(target_rows['amount'].values[0])
            
            try:
                if scenario.type == 'percentage':
                    new_amount = old_amount * (1 + scenario.value / 100)
                elif scenario.type == 'fixed':
                    new_amount = old_amount + scenario.value
                elif scenario.type == 'deferral':
                    new_amount = old_amount * (1 - scenario.value / 100)
                else:
                    print(f"Unknown scenario type: {scenario.type}")
                    return []

                # Apply the change
                self.current_budget.loc[self.current_budget['subcategory'] == scenario.target_category, 'amount'] = new_amount
                print(f"Applied {scenario.value}{'%' if scenario.type == 'percentage' or scenario.type == 'deferral' else ''} change to {scenario.target_category}")

                # Create and return budget delta
                return [BudgetDelta(
                    category=scenario.target_category,
                    old_amount=old_amount,
                    new_amount=new_amount,
                    delta=new_amount - old_amount
                )]
            except Exception as e:
                print(f"Error applying scenario: {str(e)}")
                return []
        else:
            print("No budget data available")
            return []

    def apply_multiple_scenarios(self, scenarios: List[Scenario]) -> List[BudgetDelta]:
        """Apply multiple scenarios and return all budget changes."""
        all_deltas = []
        for scenario in scenarios:
            deltas = self.apply_scenario(scenario)
            all_deltas.extend(deltas)
        return all_deltas

    def get_budget_delta(self) -> Dict[str, float]:
        """Get the difference between original and current budget."""
        return {
            category: self.current_budget[self.current_budget['subcategory'] == category]['amount'].values[0]
            for category in self.current_budget['subcategory'].tolist()
        }

    def verify_changes(self) -> bool:
        """Verify that the current budget differs from the original."""
        return not self.current_budget.empty 