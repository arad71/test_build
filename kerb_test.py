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

class EnhancedDrivewayAnalyzer:
    def __init__(self, google_api_key):
        self.api_key = google_api_key
        self.geolocator = Nominatim(user_agent="coord_analyzer")
        
        # Initialize models
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # WA-specific parameters
        self.STANDARD_CROSSOVER_WIDTH = 3.0  # meters (typical WA driveway entrance)
        self.CAMERA_HEIGHT = 2.5  # meters (Street View camera height)

    def address_to_coords(self, address):
        """Convert address to lat/long with WA normalization"""
        location = self.geolocator.geocode(f"{address}, Western Australia, Australia")
        if location:
            return (location.latitude, location.longitude)
        raise Exception(f"Geocoding failed for: {address}")

    def get_best_streetview(self, coords):
        """Get optimal Street View image with metadata"""
        params = {
            "size": "1200x800",
            "location": f"{coords[0]},{coords[1]}",
            "key": self.api_key,
            "source": "outdoor",
            "radius": 50,
            "return_error_code": True
        }
        
        # Try multiple headings to find best view
        for heading in [0, 90, 180, 270]:
            params["heading"] = heading
            response = requests.get(
                "https://maps.googleapis.com/maps/api/streetview",
                params=params
            )
            
            if response.status_code == 200:
                metadata = requests.get(
                    "https://maps.googleapis.com/maps/api/streetview/metadata",
                    params={"location": f"{coords[0]},{coords[1]}", "key": self.api_key}
                ).json()
                
                return {
                    "image": Image.open(io.BytesIO(response.content)),
                    "metadata": metadata,
                    "heading": heading
                }
        return None

    def pixel_to_world_coords(self, pixel_x, pixel_y, img_size, cam_coords, heading, pitch=0):
        """
        Convert pixel position to geographic coordinates
        Args:
            pixel_x, pixel_y: Pixel coordinates
            img_size: (width, height) of image
            cam_coords: (lat, lng) of camera
            heading: Camera heading in degrees
            pitch: Camera pitch in degrees
        Returns:
            (lat, lng) of the point in world coordinates
        """
        # Normalize pixel coordinates (-1 to 1)
        x_normalized = (2 * pixel_x / img_size[0]) - 1
        y_normalized = 1 - (2 * pixel_y / img_size[1])
        
        # Calculate angles (simplified projection)
        fov = 75  # degrees
        azimuth = heading + (x_normalized * fov/2)
        elevation = pitch + (y_normalized * fov/2)
        
        # Estimate distance (more accurate than previous version)
        distance = self.CAMERA_HEIGHT / np.tan(np.radians(abs(elevation)))
        
        # Calculate world coordinates
        return geodesic(meters=distance).destination(
            point=cam_coords, bearing=azimuth % 360
        )

    def analyze_driveway(self, address):
        """Complete analysis pipeline with coordinate output"""
        try:
            # 1. Get coordinates
            property_coords = self.address_to_coords(address)
            print(f"Property Coordinates: {property_coords}")
            
            # 2. Get best Street View
            streetview = self.get_best_streetview(property_coords)
            if not streetview:
                raise Exception("No suitable Street View found")
                
            image = streetview["image"]
            metadata = streetview["metadata"]
            heading = streetview["heading"]
            
            # 3. Detect features
            detections = self.detect_features(image)
            
            # 4. Process detections
            results = []
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            for dw in detections["driveways"]:
                dw_box = dw["box"]
                dw_center_px = ((dw_box[0] + dw_box[2]) // 2, dw_box[3])
                
                # Find nearest kerb below driveway
                nearest_kerb = min(
                    [k for k in detections["kerbs"] if k["box"][3] > dw_center_px[1]],
                    key=lambda k: abs((k["box"][0]+k["box"][2])//2 - dw_center_px[0]),
                    default=None
                )
                
                if nearest_kerb:
                    kb_box = nearest_kerb["box"]
                    kb_center_px = ((kb_box[0] + kb_box[2]) // 2, kb_box[1])
                    
                    # Calculate world coordinates
                    cam_coords = (metadata["location"]["lat"], metadata["location"]["lng"])
                    dw_coords = self.pixel_to_world_coords(
                        dw_center_px[0], dw_center_px[1],
                        image.size, cam_coords, heading
                    )
                    kb_coords = self.pixel_to_world_coords(
                        kb_center_px[0], kb_center_px[1],
                        image.size, cam_coords, heading
                    )
                    
                    # Draw visualization
                    cv2.rectangle(img_cv, (dw_box[0], dw_box[1]), (dw_box[2], dw_box[3]), (0, 255, 0), 2)
                    cv2.rectangle(img_cv, (kb_box[0], kb_box[1]), (kb_box[2], kb_box[3]), (255, 0, 0), 2)
                    cv2.line(img_cv, dw_center_px, kb_center_px, (0, 0, 255), 2)
                    
                    # Add coordinate labels
                    label_y_offset = 20
                    cv2.putText(img_cv, f"Driveway: {dw_coords.latitude:.6f}, {dw_coords.longitude:.6f}",
                                (dw_box[0], dw_box[1]-label_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(img_cv, f"Kerb: {kb_coords.latitude:.6f}, {kb_coords.longitude:.6f}",
                                (kb_box[0], kb_box[3]+label_y_offset+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Calculate distance
                    distance_m = geodesic(
                        (dw_coords.latitude, dw_coords.longitude),
                        (kb_coords.latitude, kb_coords.longitude)
                    ).meters
                    
                    results.append({
                        "driveway_px": dw_center_px,
                        "kerb_px": kb_center_px,
                        "driveway_coords": (dw_coords.latitude, dw_coords.longitude),
                        "kerb_coords": (kb_coords.latitude, kb_coords.longitude),
                        "distance_m": distance_m,
                        "heading": heading
                    })
            
            # Display results
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            plt.title(f"Driveway Analysis - {address}", pad=20)
            plt.axis('off')
            
            # Print coordinate results
            print("\nAnalysis Results:")
            for i, result in enumerate(results):
                print(f"\nFeature Pair {i+1}:")
                print(f"Driveway Coordinates: {result['driveway_coords'][0]:.6f}, {result['driveway_coords'][1]:.6f}")
                print(f"Kerb Coordinates: {result['kerb_coords'][0]:.6f}, {result['kerb_coords'][1]:.6f}")
                print(f"Physical Distance: {result['distance_m']:.2f} meters")
                print(f"View Heading: {result['heading']}°")
            
            return {
                "image": cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
                "results": results,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return None

    def detect_features(self, image):
        """Detect driveways and kerbs with confidence"""
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
                
                if "driveway" in label_name or "crossover" in label_name:
                    detections["driveways"].append({"box": box, "score": float(score)})
                elif "kerb" in label_name or "curb" in label_name:
                    detections["kerbs"].append({"box": box, "score": float(score)})
        return detections

# Example usage
if __name__ == "__main__":
    GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY"  # Replace with your key
    
    analyzer = EnhancedDrivewayAnalyzer(GOOGLE_API_KEY)
    analysis = analyzer.analyze_driveway("9 Jamaica Crossing, Langford WA")
    
    if analysis:
        # Save results to file
        with open("driveway_coordinates.txt", "w") as f:
            for i, result in enumerate(analysis["results"]):
                f.write(f"Feature Pair {i+1}:\n")
                f.write(f"Driveway: {result['driveway_coords'][0]}, {result['driveway_coords'][1]}\n")
                f.write(f"Kerb: {result['kerb_coords'][0]}, {result['kerb_coords'][1]}\n")
                f.write(f"Distance: {result['distance_m']:.2f}m\n\n")
        print("\nResults saved to 'driveway_coordinates.txt'")
