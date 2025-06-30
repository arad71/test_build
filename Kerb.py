import requests
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor

class DrivewayElevationAnalyzer:
    def __init__(self, google_api_key):
        self.api_key = google_api_key
        self.geolocator = Nominatim(user_agent="elevation_analyzer")
        
        # Initialize models
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Measurement parameters
        self.CAMERA_HEIGHT = 2.5  # meters
        self.STANDARD_CROSSOVER_WIDTH = 3.0  # meters

    def get_elevation(self, coords):
        """Get elevation data from Google Elevation API"""
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {
            "locations": f"{coords[0]},{coords[1]}",
            "key": self.api_key
        }
        response = requests.get(url, params=params).json()
        if response.get("results"):
            return response["results"][0]["elevation"]
        return None

    def create_polygon_and_analyze(self, address):
        """Full analysis pipeline with elevation data"""
        try:
            # 1. Get property coordinates
            property_coords = self.address_to_coords(address)
            
            # 2. Get Street View data
            streetview = self.get_best_streetview(property_coords)
            if not streetview:
                raise Exception("No suitable Street View found")
                
            image = streetview["image"]
            metadata = streetview["metadata"]
            heading = streetview["heading"]
            
            # 3. Detect features
            detections = self.detect_features(image)
            
            # 4. Process first driveway-kerb pair
            if not detections["driveways"] or not detections["kerbs"]:
                raise Exception("No driveway/kerb detected")
                
            dw = detections["driveways"][0]
            kb = min(
                [k for k in detections["kerbs"] if k["box"][3] > dw["box"][3]],
                key=lambda k: abs((k["box"][0]+k["box"][2])//2 - (dw["box"][0]+dw["box"][2])//2)
            )
            
            # Calculate coordinates
            cam_coords = (metadata["location"]["lat"], metadata["location"]["lng"])
            dw_coords = self.pixel_to_world_coords(
                (dw["box"][0]+dw["box"][2])//2, dw["box"][3],
                image.size, cam_coords, heading
            )
            kb_coords = self.pixel_to_world_coords(
                (kb["box"][0]+kb["box"][2])//2, kb["box"][1],
                image.size, cam_coords, heading
            )
            
            # Create point 4m right of kerb (perpendicular to driveway-kerb line)
            azimuth = geodesic(dw_coords, kb_coords).azimuth
            right_azimuth = (azimuth + 90) % 360
            right_coords = geodesic(meters=4).destination(
                point=kb_coords, bearing=right_azimuth
            )
            
            # Get elevation data
            points = [dw_coords, kb_coords, right_coords]
            elevations = []
            for point in points:
                elev = self.get_elevation((point.latitude, point.longitude))
                if elev is None:
                    raise Exception("Elevation API failed")
                elevations.append(elev)
            
            # Calculate slope
            dw_kb_distance = geodesic(
                (dw_coords.latitude, dw_coords.longitude),
                (kb_coords.latitude, kb_coords.longitude)
            ).meters
            elevation_diff = elevations[0] - elevations[1]
            slope_pct = (elevation_diff / dw_kb_distance) * 100
            
            # Visualization
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Draw driveway and kerb
            cv2.rectangle(img_cv, (dw["box"][0], dw["box"][1]), (dw["box"][2], dw["box"][3]), (0, 255, 0), 2)
            cv2.rectangle(img_cv, (kb["box"][0], kb["box"][1]), (kb["box"][2], kb["box"][3]), (255, 0, 0), 2)
            
            # Draw polygon points on image
            dw_point = ((dw["box"][0]+dw["box"][2])//2, dw["box"][3])
            kb_point = ((kb["box"][0]+kb["box"][2])//2, kb["box"][1])
            
            # Calculate right point in image coordinates
            right_vector = np.array([kb_point[0]-dw_point[0], kb_point[1]-dw_point[1]])
            right_vector = right_vector / np.linalg.norm(right_vector)
            right_vector = np.array([-right_vector[1], right_vector[0]])  # Perpendicular
            right_px = (int(kb_point[0] + right_vector[0]*100),  # Scaled for visibility
                       int(kb_point[1] + right_vector[1]*100))
            
            # Draw polygon
            polygon_pts = np.array([dw_point, kb_point, right_px])
            cv2.fillPoly(img_cv, [polygon_pts], (0, 255, 255, 50))
            cv2.polylines(img_cv, [polygon_pts], isClosed=True, color=(0, 165, 255), thickness=2)
            
            # Add labels
            cv2.putText(img_cv, f"Driveway: {elevations[0]:.1f}m", 
                        (dw["box"][0], dw["box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_cv, f"Kerb: {elevations[1]:.1f}m", 
                        (kb["box"][0], kb["box"][3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(img_cv, f"Slope: {slope_pct:.1f}%", 
                        (img_cv.shape[1]//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display results
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            plt.title(f"Driveway Elevation Analysis - {address}", pad=20)
            plt.axis('off')
            
            # Print results
            print("\n=== Analysis Results ===")
            print(f"Driveway Point: {dw_coords.latitude:.6f}, {dw_coords.longitude:.6f}")
            print(f"Kerb Point: {kb_coords.latitude:.6f}, {kb_coords.longitude:.6f}")
            print(f"Right Point (4m): {right_coords.latitude:.6f}, {right_coords.longitude:.6f}")
            print(f"\nElevation Data:")
            print(f"- Driveway: {elevations[0]:.2f} meters")
            print(f"- Kerb: {elevations[1]:.2f} meters")
            print(f"- Right Point: {elevations[2]:.2f} meters")
            print(f"\nSlope Analysis:")
            print(f"- Elevation Change: {elevation_diff:.2f} meters")
            print(f"- Horizontal Distance: {dw_kb_distance:.2f} meters")
            print(f"- Slope: {slope_pct:.1f}%")
            
            return {
                "driveway_coords": (dw_coords.latitude, dw_coords.longitude),
                "kerb_coords": (kb_coords.latitude, kb_coords.longitude),
                "right_coords": (right_coords.latitude, right_coords.longitude),
                "elevations": elevations,
                "slope_pct": slope_pct,
                "visualization": cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return None

    # [Previous methods (address_to_coords, get_best_streetview, pixel_to_world_coords, detect_features) would be here]
    # ...

# Example usage
if __name__ == "__main__":
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with actual key
    
    analyzer = DrivewayElevationAnalyzer(GOOGLE_API_KEY)
    results = analyzer.create_polygon_and_analyze("9 Jamaica Crossing, Langford WA")
    
    if results:
        # Save results to KML file
        with open("driveway_elevation.kml", "w") as f:
            f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <Placemark>
        <name>Driveway Point</name>
        <Point>
            <coordinates>{results['kerb_coords'][1]},{results['kerb_coords'][0]},0</coordinates>
        </Point>
    </Placemark>
    <Placemark>
        <name>Kerb Point</name>
        <Point>
            <coordinates>{results['driveway_coords'][1]},{results['driveway_coords'][0]},0</coordinates>
        </Point>
    </Placemark>
    <Placemark>
        <name>Right Point (4m)</name>
        <Point>
            <coordinates>{results['right_coords'][1]},{results['right_coords'][0]},0</coordinates>
        </Point>
    </Placemark>
    <Placemark>
        <name>Analysis Polygon</name>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {results['driveway_coords'][1]},{results['driveway_coords'][0]},0
                        {results['kerb_coords'][1]},{results['kerb_coords'][0]},0
                        {results['right_coords'][1]},{results['right_coords'][0]},0
                        {results['driveway_coords'][1]},{results['driveway_coords'][0]},0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
</Document>
</kml>""")
        print("\nKML file saved as 'driveway_elevation.kml'")
