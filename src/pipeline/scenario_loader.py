import json
from typing import List, Dict, Optional
from ..models.data_models import Scenario, ValidationResult, FundingConstraint

class ScenarioLoader:
    def __init__(self, funding_constraints_path: str, scenarios_path: str = None):
        """Initialize with paths to funding constraints and scenarios."""
        self.funding_constraints = self._load_funding_constraints(funding_constraints_path)
        self.scenarios_path = scenarios_path

    def _load_funding_constraints(self, path: str) -> FundingConstraint:
        """Load funding constraints from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Combine all categories and determine if any are locked
            all_categories = []
            locked_categories = set()
            notes = []
            
            for grant, details in data.items():
                categories = details.get('categories', [])
                all_categories.extend(categories)
                
                # If the grant is locked, all its categories are locked
                if details.get('locked', False):
                    locked_categories.update(categories)
                
                if details.get('note'):
                    notes.append(f"{grant}: {details['note']}")
            
            return FundingConstraint(
                categories=list(set(all_categories)),  # Remove duplicates
                locked_categories=list(locked_categories),
                note="; ".join(notes)
            )
        except Exception as e:
            print(f"Error loading funding constraints: {str(e)}")
            return FundingConstraint(categories=[], locked_categories=[], note="Error loading constraints")

    def _convert_scenario_format(self, scenario_data: Dict) -> Dict:
        """Convert old scenario format to new format."""
        try:
            new_format = {
                'id': scenario_data['id'],
                'target_category': scenario_data['target_category'],
                'description': scenario_data.get('description', '')
            }

            # Convert old format to new format
            if 'percentage' in scenario_data and scenario_data['percentage'] is not None:
                new_format['type'] = 'percentage'
                new_format['value'] = scenario_data['percentage']
            elif 'fixed_delta' in scenario_data and scenario_data['fixed_delta'] is not None:
                new_format['type'] = 'fixed'
                new_format['value'] = scenario_data['fixed_delta']
            elif 'defer_months' in scenario_data and scenario_data['defer_months'] is not None:
                new_format['type'] = 'deferral'
                new_format['value'] = scenario_data['defer_months'] * 10  # Convert months to percentage
            else:
                raise ValueError(f"Invalid scenario format: {scenario_data}")

            return new_format
        except KeyError as e:
            raise ValueError(f"Missing required field in scenario: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error converting scenario format: {str(e)}")

    def load_scenarios(self, path: str) -> List[Scenario]:
        """Load scenarios from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
                # Handle both old and new formats
                if isinstance(data, dict):
                    # New format with 'scenarios' key
                    scenarios_data = data.get('scenarios', [])
                else:
                    # Old format with direct list of scenarios
                    scenarios_data = data
                
                # Convert each scenario to the new format
                converted_scenarios = []
                for scenario in scenarios_data:
                    try:
                        converted = self._convert_scenario_format(scenario)
                        converted_scenarios.append(Scenario(**converted))
                    except Exception as e:
                        print(f"Error converting scenario {scenario.get('id', 'unknown')}: {str(e)}")
                        continue
                
                if not converted_scenarios:
                    print("Warning: No valid scenarios were loaded")
                
                return converted_scenarios
        except Exception as e:
            print(f"Error loading scenarios: {str(e)}")
            return []

    def validate_scenario(self, scenario: Scenario) -> bool:
        """Validate a scenario against funding constraints."""
        try:
            # Check if the target category exists in funding constraints
            if scenario.target_category not in self.funding_constraints.categories:
                print(f"Target category '{scenario.target_category}' not found in funding constraints")
                return False
            
            # Check if the category is locked
            if scenario.target_category in self.funding_constraints.locked_categories:
                print(f"Category '{scenario.target_category}' is locked")
                return False
            
            # Validate the scenario type and value
            if scenario.type not in ['percentage', 'fixed', 'deferral']:
                print(f"Invalid scenario type: {scenario.type}")
                return False
            
            if scenario.type == 'percentage' and not (0 <= scenario.value <= 1):
                print(f"Invalid percentage value: {scenario.value}")
                return False
            
            return True
        
        except Exception as e:
            print(f"Error validating scenario: {str(e)}")
            return False

    def load_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Load a scenario by ID."""
        try:
            if not self.scenarios_path:
                print("No scenarios path set")
                return None
            
            with open(self.scenarios_path, 'r') as f:
                scenarios = json.load(f)
                
            for scenario_data in scenarios:
                if scenario_data.get('id') == scenario_id:
                    return Scenario(**scenario_data)
                
            print(f"Scenario {scenario_id} not found")
            return None
        
        except Exception as e:
            print(f"Error loading scenario {scenario_id}: {str(e)}")
            return None

    def get_scenario_ids(self) -> List[str]:
        """Get list of all scenario IDs."""
        try:
            if not self.scenarios_path:
                print("No scenarios path set")
                return []
            
            with open(self.scenarios_path, 'r') as f:
                scenarios = json.load(f)
                return [s.get('id') for s in scenarios if s.get('id')]
            
        except Exception as e:
            print(f"Error getting scenario IDs: {str(e)}")
            return [] 