import PyPDF2
import re
import pandas as pd
from geopy.distance import distance
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

class WASitePlanAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the analyzer with a site plan PDF file
        :param file_path: Path to the PDF site plan document
        """
        self.file_path = file_path
        self.text_content = ""
        self.lot_details = {}
        self.boundary_coordinates = []
        self.extracted_data = {}
        
    def extract_text_from_pdf(self):
        """
        Extract text content from the PDF file
        """
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    self.text_content += page.extract_text()
            return True
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return False
    
    def extract_lot_details(self):
        """
        Extract lot information common in WA site plans
        """
        # Common WA lot number patterns (e.g., Lot 123, DP 456789)
        lot_pattern = r'(Lot|LOT)\s*(\d+)[,\s]*(DP|Diagram|Diagram\s*Deposited\s*Plan|Deposited\s*Plan)\s*(\d+)'
        matches = re.finditer(lot_pattern, self.text_content, re.IGNORECASE)
        
        for match in matches:
            lot_num = match.group(2)
            dp_num = match.group(4)
            self.lot_details = {
                'lot_number': lot_num,
                'dp_number': dp_num,
                'full_reference': f"Lot {lot_num} DP {dp_num}"
            }
            break  # Typically only one main lot per site plan
            
        return bool(self.lot_details)
    
    def extract_boundary_measurements(self):
        """
        Extract boundary measurements (common format in WA site plans)
        """
        # WA site plans often show boundaries with bearings and distances
        boundary_pattern = r'([NS]\s*\d+\s*\°\s*\d+\s*\′\s*\d+\s*\″\s*[EW])\s*(\d+\.\d+\s*m)'
        matches = re.finditer(boundary_pattern, self.text_content)
        
        boundaries = []
        for match in matches:
            bearing = match.group(1).replace(' ', '')
            distance = match.group(2).replace(' ', '')
            boundaries.append((bearing, distance))
            
        if boundaries:
            self.extracted_data['boundaries'] = boundaries
            return True
        return False
    
    def extract_setback_requirements(self):
        """
        Extract setback requirements which are crucial for WA developments
        """
        # Common WA setback terminology
        setback_pattern = r'(Front|Side|Rear)\s*Setback\s*:\s*(\d+\.?\d*)\s*m'
        matches = re.finditer(setback_pattern, self.text_content, re.IGNORECASE)
        
        setbacks = {}
        for match in matches:
            position = match.group(1).lower()
            distance = float(match.group(2))
            setbacks[position] = distance
            
        if setbacks:
            self.extracted_data['setbacks'] = setbacks
            return True
        return False
    
    def extract_wa_planning_codes(self):
        """
        Extract references to WA planning codes (like R-Codes)
        """
        # WA specific planning code references
        code_pattern = r'(R\-?Codes|Local\s*Planning\s*Scheme\s*\d+|LPS\d+|State\s*Planning\s*Policy\s*\d+\.\d+)'
        matches = re.finditer(code_pattern, self.text_content, re.IGNORECASE)
        
        codes = [match.group() for match in matches]
        if codes:
            self.extracted_data['planning_codes'] = list(set(codes))  # Remove duplicates
            return True
        return False
    
    def calculate_site_area(self):
        """
        Calculate site area from boundary coordinates (if available)
        """
        if not self.boundary_coordinates:
            return False
            
        try:
            polygon = Polygon(self.boundary_coordinates)
            area = polygon.area
            self.extracted_data['site_area'] = {
                'square_meters': area,
                'hectares': area / 10000,
                'acres': area / 4046.86
            }
            return True
        except Exception as e:
            print(f"Error calculating area: {e}")
            return False
    
    def analyze(self):
        """
        Run full analysis of the WA site plan
        """
        print(f"Analyzing WA site plan: {self.file_path}")
        
        # Step 1: Extract text from PDF
        if not self.extract_text_from_pdf():
            return False
            
        # Step 2: Extract lot details
        self.extract_lot_details()
        
        # Step 3: Extract boundary information
        self.extract_boundary_measurements()
        
        # Step 4: Extract WA specific planning information
        self.extract_wa_planning_codes()
        self.extract_setback_requirements()
        
        # Step 5: Calculate metrics
        self.calculate_site_area()
        
        return True
    
    def generate_report(self):
        """
        Generate a report of the analysis findings
        """
        report = {
            'file': self.file_path,
            'lot_details': self.lot_details,
            'extracted_data': self.extracted_data,
            'summary': self.create_summary()
        }
        return report
    
    def create_summary(self):
        """
        Create a human-readable summary of the analysis
        """
        summary = []
        
        if self.lot_details:
            summary.append(f"Property Identification: {self.lot_details['full_reference']}")
        
        if 'site_area' in self.extracted_data:
            area = self.extracted_data['site_area']
            summary.append(f"Site Area: {area['square_meters']:.2f} m² ({area['hectares']:.4f} ha)")
        
        if 'boundaries' in self.extracted_data:
            summary.append(f"Boundary Measurements: {len(self.extracted_data['boundaries'])} sides identified")
        
        if 'setbacks' in self.extracted_data:
            setbacks = self.extracted_data['setbacks']
            setback_info = ", ".join([f"{k}: {v}m" for k, v in setbacks.items()])
            summary.append(f"Setback Requirements: {setback_info}")
        
        if 'planning_codes' in self.extracted_data:
            codes = ", ".join(self.extracted_data['planning_codes'])
            summary.append(f"Applicable Planning Codes: {codes}")
        
        return "\n".join(summary) if summary else "No significant information extracted"
    
    def plot_boundaries(self):
        """
        Create a simple plot of the property boundaries (if coordinates are available)
        """
        if not self.boundary_coordinates:
            print("No boundary coordinates available for plotting")
            return
            
        polygon = Polygon(self.boundary_coordinates)
        x, y = polygon.exterior.xy
        
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, 'b-', linewidth=2)
        plt.fill(x, y, alpha=0.3)
        plt.title(f"Site Plan Boundary - {self.lot_details.get('full_reference', '')}")
        plt.xlabel("East-West (m)")
        plt.ylabel("North-South (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Replace with your actual WA site plan PDF path
    pdf_path = "wa_site_plan.pdf"
    
    analyzer = WASitePlanAnalyzer(pdf_path)
    if analyzer.analyze():
        report = analyzer.generate_report()
        print("\n=== Site Plan Analysis Report ===")
        print(report['summary'])
        
        # Save full report to CSV
        df = pd.DataFrame.from_dict(report, orient='index')
        df.to_csv("site_plan_analysis_report.csv")
        print("\nFull report saved to 'site_plan_analysis_report.csv'")
        
        # Plot boundaries if available
        analyzer.plot_boundaries()
    else:
        print("Failed to analyze the site plan document")
