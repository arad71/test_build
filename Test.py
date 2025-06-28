# AI Models Setup and Installation Guide

"""
STEP 1: Install Required Packages
Run these commands in your terminal:
"""

# Basic requirements
install_commands = """
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install easyocr
pip install opencv-python
pip install pillow
pip install numpy
pip install PyMuPDF
pip install pandas
"""

"""
STEP 2: Alternative Lightweight AI Setup
If the full setup is too heavy, use this lightweight version:
"""

import cv2
import numpy as np
from PIL import Image
import easyocr
import fitz
import re
import json

class LightweightAISitePlanReader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.extracted_data = {}
        
        # Initialize lightweight AI models
        print("Loading lightweight AI models...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU version
        print("AI models ready!")
    
    def extract_images(self, dpi=300):
        """Extract high-quality images from PDF"""
        images = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img)
        
        return images
    
    def ai_text_extraction(self, images):
        """Extract text using EasyOCR AI model"""
        text_results = []
        
        for i, img in enumerate(images):
            print(f"Processing page {i+1} with AI OCR...")
            
            # Use EasyOCR for text detection
            ocr_results = self.easyocr_reader.readtext(img)
            
            # Process results
            page_text = []
            full_text = ""
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:  # Filter low confidence
                    # Calculate bounding box center
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    
                    page_text.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': (center_x, center_y)
                    })
                    
                    full_text += text + " "
            
            text_results.append({
                'page': i,
                'elements': page_text,
                'full_text': full_text.strip()
            })
        
        self.extracted_data['ai_text'] = text_results
        return text_results
    
    def smart_element_detection(self, text_results):
        """Detect site plan elements using text analysis"""
        
        # Define element keywords for different categories
        element_categories = {
            'drainage': [
                'drain', 'drainage', 'storm water', 'stormwater', 'soak well', 
                'pipe', 'plumbing', 'water', 'flood', 'runoff'
            ],
            'boundaries': [
                'boundary', 'property line', 'fence', 'border', 'lot line'
            ],
            'structures': [
                'house', 'building', 'garage', 'shed', 'concrete', 'driveway',
                'crossover', 'footpath', 'road'
            ],
            'measurements': [
                'm', 'meter', 'metre', 'feet', 'ft', 'inch', 'mm', 'cm',
                'approx', 'approximately'
            ],
            'permissions': [
                'council', 'permit', 'approval', 'permission', 'application',
                'proposed', 'existing'
            ]
        }
        
        detected_elements = []
        
        for page_data in text_results:
            page_elements = {
                'page': page_data['page'],
                'categories': {category: [] for category in element_categories},
                'confidence_scores': {}
            }
            
            full_text = page_data['full_text'].lower()
            
            # Check each category
            for category, keywords in element_categories.items():
                category_score = 0
                found_keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in full_text:
                        found_keywords.append(keyword)
                        category_score += full_text.count(keyword.lower())
                
                if found_keywords:
                    page_elements['categories'][category] = found_keywords
                    page_elements['confidence_scores'][category] = category_score
            
            detected_elements.append(page_elements)
        
        self.extracted_data['smart_elements'] = detected_elements
        return detected_elements
    
    def extract_measurements_ai(self, text_results):
        """Extract measurements using AI-enhanced pattern matching"""
        
        # Enhanced measurement patterns
        patterns = {
            'metric_precise': r'\d+\.\d+\s*m(?:[^a-zA-Z]|$)',
            'metric_whole': r'\d+\s*m(?:[^a-zA-Z]|$)',
            'imperial': r'\d+\'\s*-?\s*\d*"?',
            'dimensions': r'\d+\.?\d*\s*[×xX]\s*\d+\.?\d*',
            'approximate': r'(?:approx\.?\s*|~\s*)\d+\.?\d*\s*m',
            'range': r'\d+\.?\d*\s*-\s*\d+\.?\d*\s*m',
            'decimal': r'\d+\.\d+(?=\s|$)',
            'area': r'\d+\.?\d*\s*(?:m²|sq\s*m|square\s*met)'
        }
        
        measurement_results = []
        
        for page_data in text_results:
            page_measurements = []
            
            for text_element in page_data['elements']:
                text = text_element['text']
                
                for pattern_name, pattern in patterns.items():
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Determine context using surrounding text
                        context = self._determine_context(text, page_data['full_text'])
                        
                        measurement_info = {
                            'value': match.group().strip(),
                            'type': pattern_name,
                            'context': context,
                            'source_text': text,
                            'confidence': text_element['confidence'],
                            'position': text_element['center'],
                            'bbox': text_element['bbox']
                        }
                        
                        page_measurements.append(measurement_info)
            
            measurement_results.append({
                'page': page_data['page'],
                'measurements': page_measurements
            })
        
        self.extracted_data['ai_measurements'] = measurement_results
        return measurement_results
    
    def _determine_context(self, text, full_page_text):
        """Determine the context of a measurement"""
        contexts = {
            'drainage': ['drain', 'pipe', 'water', 'storm', 'soak'],
            'boundary': ['boundary', 'property', 'line', 'fence'],
            'structure': ['house', 'building', 'concrete', 'driveway'],
            'distance': ['to road', 'distance', 'length', 'width'],
            'connection': ['connection', 'plumbing', 'council']
        }
        
        text_lower = (text + " " + full_page_text).lower()
        
        for context, keywords in contexts.items():
            if any(keyword in text_lower for keyword in keywords):
                return context
        
        return 'general'
    
    def generate_ai_summary(self):
        """Generate intelligent summary of findings"""
        summary = {
            'total_pages': len(self.doc),
            'elements_by_category': {},
            'key_measurements': [],
            'recommendations': []
        }
        
        # Summarize detected elements
        smart_elements = self.extracted_data.get('smart_elements', [])
        all_categories = {}
        
        for page_data in smart_elements:
            for category, keywords in page_data['categories'].items():
                if keywords:
                    if category not in all_categories:
                        all_categories[category] = set()
                    all_categories[category].update(keywords)
        
        summary['elements_by_category'] = {
            category: list(keywords) 
            for category, keywords in all_categories.items()
        }
        
        # Extract key measurements
        measurements = self.extracted_data.get('ai_measurements', [])
        all_measurements = []
        
        for page_data in measurements:
            for measurement in page_data['measurements']:
                if measurement['confidence'] > 0.7:
                    all_measurements.append({
                        'value': measurement['value'],
                        'context': measurement['context'],
                        'page': page_data['page'] + 1,
                        'confidence': round(measurement['confidence'], 2)
                    })
        
        # Sort by confidence and take top measurements
        summary['key_measurements'] = sorted(
            all_measurements, 
            key=lambda x: x['confidence'], 
            reverse=True
        )[:10]
        
        # Generate recommendations
        recommendations = []
        
        if 'drainage' in all_categories:
            recommendations.append("Document contains drainage elements - suitable for water management planning")
        
        if 'permissions' in all_categories:
            recommendations.append("Permit-related terms detected - likely a planning application")
        
        if len(all_measurements) > 5:
            recommendations.append(f"High measurement density detected ({len(all_measurements)} measurements)")
        
        summary['recommendations'] = recommendations
        self.extracted_data['summary'] = summary
        
        return summary
    
    def analyze_with_lightweight_ai(self):
        """Complete analysis using lightweight AI"""
        print("Starting lightweight AI analysis...")
        
        # Extract images
        print("Extracting images...")
        images = self.extract_images()
        
        # AI text extraction
        print("AI text extraction...")
        text_results = self.ai_text_extraction(images)
        
        # Smart element detection
        print("Detecting elements...")
        self.smart_element_detection(text_results)
        
        # AI measurement extraction
        print("Extracting measurements...")
        self.extract_measurements_ai(text_results)
        
        # Generate summary
        print("Generating AI summary...")
        summary = self.generate_ai_summary()
        
        print("Lightweight AI analysis complete!")
        return self.extracted_data
    
    def print_ai_results(self):
        """Print formatted AI analysis results"""
        summary = self.extracted_data.get('summary', {})
        
        print("\n" + "="*60)
        print("LIGHTWEIGHT AI SITE PLAN ANALYSIS")
        print("="*60)
        
        print(f"Total Pages: {summary.get('total_pages', 0)}")
        
        print("\nDETECTED ELEMENTS BY CATEGORY:")
        print("-" * 40)
        elements = summary.get('elements_by_category', {})
        for category, keywords in elements.items():
            print(f"{category.upper()}: {', '.join(keywords)}")
        
        print("\nKEY MEASUREMENTS (AI Confidence):")
        print("-" * 40)
        measurements = summary.get('key_measurements', [])
        for measurement in measurements:
            print(f"• {measurement['value']} ({measurement['context']}) "
                  f"- Page {measurement['page']}, Conf: {measurement['confidence']}")
        
        print("\nAI RECOMMENDATIONS:")
        print("-" * 40)
        recommendations = summary.get('recommendations', [])
        for rec in recommendations:
            print(f"• {rec}")
    
    def save_results(self, filename="lightweight_ai_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.extracted_data, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def close(self):
        """Close PDF document"""
        self.doc.close()


# Simple usage example
def analyze_pdf_with_lightweight_ai(pdf_path):
    """Analyze PDF using lightweight AI models"""
    try:
        # Initialize reader
        reader = LightweightAISitePlanReader(pdf_path)
        
        # Run analysis
        reader.analyze_with_lightweight_ai()
        
        # Print results
        reader.print_ai_results()
        
        # Save results
        reader.save_results()
        
        # Close
        reader.close()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")

# For your drainage proposal PDF
if __name__ == "__main__":
    # Simple installation check
    try:
        import easyocr
        print("✓ EasyOCR available")
        
        # Analyze your drainage PDF
        pdf_path = "drainage_proposal.pdf"  # Update with your file path
        analyze_pdf_with_lightweight_ai(pdf_path)
        
    except ImportError:
        print("Please install EasyOCR first:")
        print("pip install easyocr")
