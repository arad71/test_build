import json
from typing import Dict, List, Optional

class WABuildingRegulations:
    """Western Australia specific building regulations"""
    # Minimum dimensions and areas
    MIN_ROOM_HEIGHT = 2.4  # meters
    MIN_HABITABLE_ROOM_AREA = 6.0  # m²
    MIN_HABITABLE_ROOM_WIDTH = 2.4  # meters
    MIN_BEDROOM_AREA = 6.0  # m² (must have at least one bedroom ≥ 9.5m²)
    MIN_MASTER_BEDROOM_AREA = 9.5  # m²
    MIN_BATHROOM_AREA = 1.2  # m² (for half bath)
    MIN_FULL_BATHROOM_AREA = 3.0  # m²
    MIN_KITCHEN_AREA = 4.0  # m²
    MIN_LIVING_AREA = 12.0  # m²
    MIN_STAIR_WIDTH = 0.6  # meters
    MIN_DOOR_WIDTH = 0.6  # meters (0.8m recommended for accessibility)
    MIN_WINDOW_AREA = 0.1  # m² (for ventilation)
    WINDOW_TO_FLOOR_RATIO = 0.1  # 10% of floor area for natural light
    
    # Energy efficiency requirements (WA specific)
    MAX_EXTERNAL_WALL_AREA = 0.5  # Max 50% of total wall area can be glass
    MIN_INSULATION_RVALUES = {
        'ceiling': 2.8,
        'external_wall': 2.2,
        'floor': 1.3
    }

class FloorPlanValidator:
    def __init__(self):
        self.regulations = WABuildingRegulations()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.floor_plan: Optional[Dict] = None
    
    def load_design_document(self, file_path: str) -> bool:
        """Load design document from JSON file"""
        try:
            with open(file_path, 'r') as f:
                self.floor_plan = json.load(f)
            return True
        except Exception as e:
            self.errors.append(f"Failed to load design document: {str(e)}")
            return False
    
    def validate_room(self, room: Dict) -> None:
        """Validate a single room against WA regulations"""
        room_type = room['type'].lower()
        dims = room['dimensions']
        area = dims['length'] * dims['width']
        min_dim = min(dims['length'], dims['width'])
        
        # Common checks for all rooms
        if dims.get('height', 2.4) < self.regulations.MIN_ROOM_HEIGHT:
            self.errors.append(f"{room_type.capitalize()} height {dims['height']:.2f}m is below minimum {self.regulations.MIN_ROOM_HEIGHT}m")
        
        # Room type specific checks
        if room_type == 'bedroom':
            if 'master' in room.get('tags', []) and area < self.regulations.MIN_MASTER_BEDROOM_AREA:
                self.errors.append(f"Master bedroom area {area:.2f}m² is below minimum {self.regulations.MIN_MASTER_BEDROOM_AREA}m²")
            elif area < self.regulations.MIN_BEDROOM_AREA:
                self.errors.append(f"Bedroom area {area:.2f}m² is below minimum {self.regulations.MIN_BEDROOM_AREA}m²")
            
            if min_dim < self.regulations.MIN_HABITABLE_ROOM_WIDTH:
                self.errors.append(f"Bedroom width {min_dim:.2f}m is below minimum {self.regulations.MIN_HABITABLE_ROOM_WIDTH}m")
        
        elif room_type == 'bathroom':
            min_area = self.regulations.MIN_FULL_BATHROOM_AREA if 'shower' in room.get('features', []) else self.regulations.MIN_BATHROOM_AREA
            if area < min_area:
                self.errors.append(f"{room_type.capitalize()} area {area:.2f}m² is below minimum {min_area}m²")
        
        elif room_type == 'kitchen':
            if area < self.regulations.MIN_KITCHEN_AREA:
                self.errors.append(f"Kitchen area {area:.2f}m² is below minimum {self.regulations.MIN_KITCHEN_AREA}m²")
        
        elif room_type in ['living', 'dining', 'family']:
            if area < self.regulations.MIN_LIVING_AREA:
                self.errors.append(f"{room_type.capitalize()} area {area:.2f}m² is below minimum {self.regulations.MIN_LIVING_AREA}m²")
        
        # Window and ventilation checks
        window_area = room.get('window_area', 0)
        if window_area < self.regulations.MIN_WINDOW_AREA:
            self.warnings.append(f"{room_type.capitalize()} window area {window_area:.2f}m² is below minimum {self.regulations.MIN_WINDOW_AREA}m²")
        
        window_ratio = window_area / area if area > 0 else 0
        if window_ratio < self.regulations.WINDOW_TO_FLOOR_RATIO:
            self.warnings.append(f"{room_type.capitalize()} window to floor ratio ({window_ratio:.1%}) is below recommended {self.regulations.WINDOW_TO_FLOOR_RATIO:.0%}")
    
    def validate_energy_efficiency(self) -> None:
        """Check WA specific energy efficiency requirements"""
        if not self.floor_plan or 'construction' not in self.floor_plan:
            return
            
        constr = self.floor_plan['construction']
        
        # Check insulation values
        for element, min_rvalue in self.regulations.MIN_INSULATION_RVALUES.items():
            if element in constr.get('insulation', {}):
                if constr['insulation'][element] < min_rvalue:
                    self.errors.append(f"{element.replace('_', ' ').capitalize()} insulation R-value {constr['insulation'][element]} is below minimum {min_rvalue}")
        
        # Check glazing percentage
        total_wall_area = constr.get('external_wall_area', 0)
        glass_area = constr.get('glass_area', 0)
        if total_wall_area > 0:
            glass_ratio = glass_area / total_wall_area
            if glass_ratio > self.regulations.MAX_EXTERNAL_WALL_AREA:
                self.warnings.append(f"Glass area ({glass_ratio:.0%} of walls) exceeds recommended maximum {self.regulations.MAX_EXTERNAL_WALL_AREA:.0%}")
    
    def validate_floor_plan(self) -> bool:
        """Validate the entire floor plan against WA regulations"""
        if not self.floor_plan:
            self.errors.append("No floor plan loaded")
            return False
        
        # Check required rooms exist
        required_rooms = {'bedroom': 1, 'bathroom': 1, 'kitchen': 1, 'living': 1}
        room_counts = {room: 0 for room in required_rooms}
        
        for room in self.floor_plan.get('rooms', []):
            room_type = room['type'].lower()
            if room_type in room_counts:
                room_counts[room_type] += 1
            self.validate_room(room)
        
        # Check we have at least the required rooms
        for room_type, min_count in required_rooms.items():
            if room_counts[room_type] < min_count:
                self.errors.append(f"Floor plan must contain at least {min_count} {room_type}")
        
        # Check doors
        for door in self.floor_plan.get('doors', []):
            if door['width'] < self.regulations.MIN_DOOR_WIDTH:
                self.errors.append(f"Door width {door['width']:.2f}m is below minimum {self.regulations.MIN_DOOR_WIDTH}m")
        
        # Check stairs
        for stair in self.floor_plan.get('stairs', []):
            if stair['width'] < self.regulations.MIN_STAIR_WIDTH:
                self.errors.append(f"Stair width {stair['width']:.2f}m is below minimum {self.regulations.MIN_STAIR_WIDTH}m")
        
        # Check energy efficiency
        self.validate_energy_efficiency()
        
        return len(self.errors) == 0
    
    def generate_report(self) -> str:
        """Generate a validation report"""
        report = []
        report.append("Western Australia Floor Plan Validation Report")
        report.append("=" * 50)
        
        if self.errors:
            report.append("\nERRORS (Must be addressed):")
            for i, error in enumerate(self.errors, 1):
                report.append(f"{i}. {error}")
        
        if self.warnings:
            report.append("\nWARNINGS (Recommended improvements):")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"{i}. {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\nNo issues found. Floor plan complies with WA regulations.")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    validator = FloorPlanValidator()
    
    # Load the submitted design document
    if validator.load_design_document("submitted_design.json"):
        # Validate the floor plan
        is_valid = validator.validate_floor_plan()
        
        # Generate and print the report
        print(validator.generate_report())
        
        # Exit with appropriate status code
        exit(0 if is_valid else 1)
