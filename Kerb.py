import requests
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor

class LangfordDrivewayAnalyzer:
    def __init__(self, google_api_key):
        self.api_key = google_api_key
        self.geolocator = Nominatim(user_agent="driveway_analysis")
        
        # Initialize model with Australian road feature awareness
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # WA-specific feature classes
        self.driveway_classes = ["driveway", "crossover", "garage_entry"]
        self.kerb_classes = ["kerb", "verge", "nature_strip"]

    def address_to_coords(self, address):
        """Convert street address to coordinates"""
        location = self.geolocator.geocode(f"{address}, Western Australia")
        if location:
            return (location.latitude, location.longitude)
        raise Exception(f"Could not geocode address: {address}")

    def get_streetview_image(self, location, heading=0, pitch=10, fov=75):
        """Get Street View image with WA-optimized parameters"""
        params = {
            "size": "1200x800",
            "location": f"{location[0]},{location[1]}",
            "heading": heading,
            "pitch": pitch,  # Downward angle for better driveway view
            "fov": fov,  # Narrower FOV for less distortion
            "key": self.api_key,
            "source": "outdoor"  # Prefer outdoor imagery
        }
        response = requests.get("https://maps.googleapis.com/maps/api/streetview", params=params)
        return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None

    def detect_wa_features(self, image):
        """Detect driveways and kerbs with WA characteristics"""
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.feature_extractor.post_process(outputs, target_sizes)[0]
        
        detections = {"driveways": [], "kerbs": []}
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.7:
                box = [int(i) for i in box.tolist()]
                label_name = self.model.config.id2label[label.item()].lower()
                
                if any(dw in label_name for dw in self.driveway_classes):
                    detections["driveways"].append({"box": box, "score": float(score), "type": "driveway"})
                elif any(kb in label_name for kb in self.kerb_classes):
                    detections["kerbs"].append({"box": box, "score": float(score), "type": "kerb"})
        return detections

    def analyze_property(self, address):
        """Full analysis pipeline for a Langford property"""
        try:
            # 1. Convert address to coordinates
            coords = self.address_to_coords(address)
            print(f"Coordinates for {address}: {coords}")
            
            # 2. Get optimal Street View (try multiple headings)
            for heading in [0, 90, 180, 270]:
                image = self.get_streetview_image(coords, heading=heading)
                if image:
                    # 3. Detect features
                    detections = self.detect_wa_features(image)
                    
                    # 4. Find driveway-kerb relationships
                    if detections["driveways"] and detections["kerbs"]:
                        return self.visualize_analysis(image, detections, address)
            
            print("No suitable driveway/kerb detected from any angle")
            return None
            
        except Exception as e:
            print(f"Error analyzing {address}: {str(e)}")
            return None

    def visualize_analysis(self, image, detections, address):
        """Create visualization with measurements"""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Process all driveways
        for dw in detections["driveways"]:
            dw_box = dw["box"]
            dw_center = ((dw_box[0] + dw_box[2]) // 2, dw_box[3])  # Bottom center
            
            # Find closest kerb below driveway
            closest_kerb = min(
                [kb for kb in detections["kerbs"] if kb["box"][3] > dw_center[1]],  # Kerbs below driveway
                key=lambda kb: abs((kb["box"][0] + kb["box"][2])//2 - dw_center[0]),
                default=None
            )
            
            if closest_kerb:
                kb_box = closest_kerb["box"]
                kb_center = ((kb_box[0] + kb_box[2]) // 2, kb_box[1])  # Top center
                
                # Draw elements
                cv2.rectangle(img_cv, (dw_box[0], dw_box[1]), (dw_box[2], dw_box[3]), (0, 255, 0), 2)
                cv2.rectangle(img_cv, (kb_box[0], kb_box[1]), (kb_box[2], kb_box[3]), (255, 0, 0), 2)
                cv2.line(img_cv, dw_center, kb_center, (0, 0, 255), 2)
                
                # Add distance measurement (using standard WA crossover width ~3m for scale)
                px_distance = np.sqrt((kb_center[0]-dw_center[0])**2 + (kb_center[1]-dw_center[1])**2)
                est_distance = round(px_distance * (3.0 / (dw_box[2]-dw_box[0])), 1)
                mid_point = ((dw_center[0]+kb_center[0])//2, (dw_center[1]+kb_center[1])//2)
                cv2.putText(img_cv, f"{est_distance}m", mid_point, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert back to RGB for matplotlib
        result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Display results
        plt.figure(figsize=(14, 8))
        plt.imshow(result_img)
        plt.title(f"Driveway Analysis\n{address}", pad=20)
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=3, label='Driveway'),
            plt.Line2D([0], [0], color='blue', lw=3, label='Kerb'),
            plt.Line2D([0], [0], color='red', lw=3, label='Distance (m)')
        ]
        plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.8, 0))
        
        plt.tight_layout()
        plt.show()
        return result_img

# Example usage
if __name__ == "__main__":
    GOOGLE_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual key
    
    analyzer = LangfordDrivewayAnalyzer(GOOGLE_API_KEY)
    analysis_result = analyzer.analyze_property("9 Jamaica Crossing, Langford WA")
    
    if analysis_result is None:
        print("Analysis failed. Try adjusting parameters or checking the address.")
