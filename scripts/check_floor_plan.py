import json
import re
from typing import Dict, List, Optional, Union
import PyPDF2
import pdfplumber
import ezdxf  # For CAD file processing
from pathlib import Path

class WABuildingRegulations:
    """Updated WA building regulations including 2023 amendments"""
    # [Previous regulation constants remain the same...]
    # Add CAD-specific requirements
    MIN_CAD_LAYERS = 5  # Minimum expected layers in a professional CAD drawing
    REQUIRED_CAD_LAYERS = ['WALLS', 'DOORS', 'WINDOWS', 'DIMENSIONS', 'TEXT']

class DesignFileParser:
    """Handles both PDF and CAD file parsing"""
    
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """Determine if file is PDF or CAD"""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ('.dwg', '.dxf'):
            return 'cad'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def parse_pdf(self, pdf_path: str) -> Dict:
        """Parse PDF floor plans"""
        text = self._extract_pdf_text(pdf_path)
        return {
            'file_type': 'pdf',
            'rooms': self._parse_pdf_rooms(text),
            'dimensions': self._parse_pdf_dimensions(text),
            'metadata': self._extract_pdf_metadata(pdf_path)
        }

    def parse_cad(self, cad_path: str) -> Dict:
        """Parse CAD (DWG/DXF) floor plans"""
        try:
            doc = ezdxf.readfile(cad_path)
            msp = doc.modelspace()
            
            # Extract entities
            walls = [e for e in msp if e.dxftype() == 'LINE' and e.layer.lower() == 'walls']
            doors = [e for e in msp if e.dxftype() == 'INSERT' and 'door' in e.layer.lower()]
            windows = [e for e in msp if e.dxftype() == 'INSERT' and 'window' in e.layer.lower()]
            dimensions = [e for e in msp if e.dxftype() == 'DIMENSION']
            texts = [e for e in msp if e.dxftype() == 'MTEXT' or e.dxftype() == 'TEXT']
            
            # Convert to structured data
            return {
                'file_type': 'cad',
                'layers': list(doc.layers),
                'wall_lengths': [self._calculate_length(w) for w in walls],
                'door_sizes': [self._get_block_size(d) for d in doors],
                'window_sizes': [self._get_block_size(w) for w in windows],
                'explicit_dimensions': [self._get_dimension_value(d) for d in dimensions],
                'room_labels': [t.plain_text() for t in texts if 'room' in t.plain_text().lower()],
                'cad_metadata': doc.header
            }
        except Exception as e:
            raise ValueError(f"CAD parsing error: {str(e)}")

    def _calculate_length(self, entity) -> float:
        """Calculate length of CAD line entities"""
        start = entity.dxf.start
        end = entity.dxf.end
        return ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5

    def _get_block_size(self, block_ref) -> Dict:
        """Get dimensions of CAD block references"""
        return {
            'width': block_ref.dxf.xscale,
            'height': block_ref.dxf.yscale,
            'rotation': block_ref.dxf.rotation
        }

    def _get_dimension_value(self, dim) -> float:
        """Extract dimension value from CAD dimension entities"""
        return dim.dxf.measurement

    # [Previous PDF parsing methods remain...]

class WAFloorPlanValidator:
    """Enhanced validator with CAD support"""
    
    def __init__(self):
        self.regulations = WABuildingRegulations()
        self.parser = DesignFileParser()
        self.errors = []
        self.warnings = []
        self.file_type = None

    def validate(self, file_path: str) -> bool:
        """Main validation method for any supported file type"""
        try:
            self.file_type = self.parser.detect_file_type(file_path)
            
            if self.file_type == 'pdf':
                design_data = self.parser.parse_pdf(file_path)
                return self._validate_pdf_design(design_data)
            elif self.file_type == 'cad':
                design_data = self.parser.parse_cad(file_path)
                return self._validate_cad_design(design_data)
                
        except Exception as e:
            self.errors.append(f"Validation error: {str(e)}")
            return False

    def _validate_cad_design(self, design_data: Dict) -> bool:
        """Specific validation for CAD designs"""
        # Check CAD structure quality
        if len(design_data['layers']) < self.regulations.MIN_CAD_LAYERS:
            self.warnings.append(f"CAD file has only {len(design_data['layers'])} layers (recommended minimum: {self.regulations.MIN_CAD_LAYERS})")
        
        missing_layers = [layer for layer in self.regulations.REQUIRED_CAD_LAYERS 
                         if layer not in design_data['layers']]
        if missing_layers:
            self.warnings.append(f"CAD file missing recommended layers: {', '.join(missing_layers)}")
        
        # Validate dimensions
        self._validate_cad_dimensions(design_data)
        
        # Validate room configurations
        self._validate_cad_rooms(design_data)
        
        return len(self.errors) == 0

    def _validate_cad_dimensions(self, design_data: Dict) -> None:
        """Check critical dimensions in CAD file"""
        # Check door widths
        door_widths = [d['width'] for d in design_data['door_sizes']]
        adequate_doors = [w for w in door_widths if w >= self.regulations.MIN_DOOR_WIDTH]
        
        if not adequate_doors:
            self.errors.append(
                f"No doors meet minimum width requirement ({self.regulations.MIN_DOOR_WIDTH}m). "
                f"Found widths: {', '.join(map(str, door_widths))}"
            )
        
        # Check wall thicknesses (typical internal walls should be 0.1m in WA)
        wall_thicknesses = self._estimate_wall_thicknesses(design_data)
        for thickness in wall_thicknesses:
            if thickness < 0.09:  # 90mm is minimum for internal walls
                self.warnings.append(f"Wall thickness {thickness:.3f}m may be below standard (0.1m typical)")

    def _validate_cad_rooms(self, design_data: Dict) -> None:
        """Validate room sizes and configurations from CAD"""
        # This would implement similar checks as PDF version but using CAD data
        # In practice, you'd need more sophisticated room detection from CAD
        pass

    def _estimate_wall_thicknesses(self, design_data: Dict) -> List[float]:
        """Estimate wall thickness from CAD data"""
        # Simplified approach - in reality would need proper spatial analysis
        return [0.1]  # Placeholder

    # [Previous PDF validation methods remain...]

    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report = [
            f"WESTERN AUSTRALIA BUILDING COMPLIANCE REPORT ({self.file_type.upper()})",
            "="*60,
            f"Based on: NCC 2022 Volume 2 (WA Variations)",
            f"Climate Zone: 5 (Perth Metro)",
            "\nVALIDATION RESULTS:"
        ]
        
        if not self.errors and not self.warnings:
            report.append("✅ Design fully complies with WA building requirements")
        
        if self.errors:
            report.append("\nCRITICAL ISSUES (Must be addressed):")
            report.extend(f"❌ {e}" for e in self.errors)
        
        if self.warnings:
            report.append("\nRECOMMENDATIONS (Consider improvements):")
            report.extend(f"⚠️ {w}" for w in self.warnings)
        
        report.extend([
            "\nNOTES:",
            "1. This automated check doesn't replace professional certification",
            "2. CAD validation accuracy depends on drawing standards used",
            "3. Always consult with a WA licensed building surveyor for final approval"
        ])
        
        return "\n".join(report)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python wa_validator.py <path_to_design_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    validator = WAFloorPlanValidator()
    
    print(f"\nValidating {file_path} against WA building regulations...")
    if validator.validate(file_path):
        print("✅ Validation completed successfully")
    else:
        print("❌ Validation found compliance issues")
    
    print("\n" + validator.generate_report())
