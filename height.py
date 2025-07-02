import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Install required packages:
# pip install torch torchvision opencv-python pillow matplotlib numpy requests

@dataclass
class GeoPoint:
    """Geographic point with latitude and longitude"""
    lat: float
    lng: float
    elevation: Optional[float] = None

@dataclass
class DetectedObject:
    """Detected object with geographic information"""
    id: int
    bbox_pixels: Tuple[int, int, int, int]  # x, y, width, height
    center_geo: GeoPoint
    area_pixels: int
    height_estimates: Dict[str, float]
    confidence: float
    elevation: Optional[float] = None

class GeoObjectHeightDetector:
    def __init__(self, elevation_api_key=None):
        """
        Initialize detector with optional elevation API key
        
        Args:
            elevation_api_key: Google Maps Elevation API key (optional)
        """
        self.elevation_api_key = elevation_api_key
        self.setup_models()
    
    def setup_models(self):
        """Initialize pre-trained models for depth estimation and object detection"""
        try:
            # Load MiDaS model for monocular depth estimation
            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.midas.eval()
            
            # MiDaS transforms
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = self.midas_transforms.small_transform
            
            print("✓ MiDaS depth estimation model loaded")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to alternative methods...")
            self.midas = None
    
    def load_image(self, image_path):
        """Load image from file path or URL"""
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def setup_georeferencing(self, image_bounds: Tuple[float, float, float, float], 
                           image_shape: Tuple[int, int]):
        """
        Setup geographic coordinate transformation
        
        Args:
            image_bounds: (north_lat, south_lat, east_lng, west_lng)
            image_shape: (height, width) of the image in pixels
        """
        self.north_lat, self.south_lat, self.east_lng, self.west_lng = image_bounds
        self.image_height, self.image_width = image_shape
        
        # Calculate scaling factors
        self.lat_range = self.north_lat - self.south_lat
        self.lng_range = self.east_lng - self.west_lng
        
        print(f"✓ Georeferencing setup: {self.lat_range:.6f}° lat x {self.lng_range:.6f}° lng")
    
    def pixel_to_geo(self, x_pixel: int, y_pixel: int) -> GeoPoint:
        """Convert pixel coordinates to geographic coordinates"""
        # Calculate lat/lng from pixel position
        lng = self.west_lng + (x_pixel / self.image_width) * self.lng_range
        lat = self.north_lat - (y_pixel / self.image_height) * self.lat_range
        
        return GeoPoint(lat=lat, lng=lng)
    
    def geo_to_pixel(self, geo_point: GeoPoint) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel coordinates"""
        x_pixel = int(((geo_point.lng - self.west_lng) / self.lng_range) * self.image_width)
        y_pixel = int(((self.north_lat - geo_point.lat) / self.lat_range) * self.image_height)
        
        return x_pixel, y_pixel
    
    def get_elevation_google(self, points: List[GeoPoint]) -> List[GeoPoint]:
        """Get elevation data using Google Elevation API"""
        if not self.elevation_api_key:
            print("⚠️ No Google API key provided for elevation data")
            return points
        
        # Prepare locations for API call
        locations = "|".join([f"{point.lat},{point.lng}" for point in points])
        
        url = f"https://maps.googleapis.com/maps/api/elevation/json"
        params = {
            'locations': locations,
            'key': self.elevation_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                for i, result in enumerate(data['results']):
                    if i < len(points):
                        points[i].elevation = result['elevation']
                print(f"✓ Retrieved elevation for {len(points)} points")
            else:
                print(f"❌ Google Elevation API error: {data['status']}")
                
        except Exception as e:
            print(f"❌ Elevation API request failed: {e}")
        
        return points
    
    def get_elevation_open(self, points: List[GeoPoint]) -> List[GeoPoint]:
        """Get elevation data using Open Elevation API (free, no key required)"""
        try:
            # Prepare locations for API call
            locations = [{"latitude": point.lat, "longitude": point.lng} for point in points]
            
            url = "https://api.open-elevation.com/api/v1/lookup"
            
            # Split into chunks of 100 points (API limit)
            chunk_size = 100
            for i in range(0, len(points), chunk_size):
                chunk_points = points[i:i+chunk_size]
                chunk_locations = locations[i:i+chunk_size]
                
                response = requests.post(url, json={"locations": chunk_locations})
                data = response.json()
                
                if 'results' in data:
                    for j, result in enumerate(data['results']):
                        if i + j < len(points):
                            points[i + j].elevation = result['elevation']
                
                # Rate limiting
                time.sleep(0.1)
            
            print(f"✓ Retrieved elevation for {len(points)} points using Open Elevation")
            
        except Exception as e:
            print(f"❌ Open Elevation API request failed: {e}")
        
        return points
    
    def estimate_depth_midas(self, image):
        """Estimate depth using MiDaS model"""
        if self.midas is None:
            raise ValueError("MiDaS model not available")
        
        # Prepare image for MiDaS
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0)
        
        # Predict depth
        with torch.no_grad():
            depth = self.midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth_map = depth.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_map
    
    def detect_objects_contour(self, image, min_area=500):
        """Detect objects using contour detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area
                })
        
        return objects
    
    def estimate_height_from_depth(self, depth_map, object_bbox, 
                                  camera_height=100, pixel_to_meter_ratio=0.1):
        """Estimate object height from depth map"""
        x, y, w, h = object_bbox
        
        # Extract depth values for the object
        object_depth = depth_map[y:y+h, x:x+w]
        
        # Calculate relative depth differences
        ground_depth = np.percentile(object_depth, 85)  # Assume ground level
        object_top_depth = np.percentile(object_depth, 15)  # Assume object top
        
        # Depth difference (in depth map units)
        depth_diff = ground_depth - object_top_depth
        
        # Convert to meters (this is an approximation and needs calibration)
        estimated_height = depth_diff * pixel_to_meter_ratio
        
        return max(0, estimated_height)
    
    def process_polygon_area(self, image, polygon_geo_points: List[GeoPoint]):
        """Process objects within a specific polygon area defined by geographic coordinates"""
        # Convert geographic polygon to pixel coordinates
        polygon_pixels = []
        for geo_point in polygon_geo_points:
            x, y = self.geo_to_pixel(geo_point)
            polygon_pixels.append([x, y])
        
        # Create mask for polygon
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygon_array = np.array(polygon_pixels, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_array], 255)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image, mask, polygon_pixels
    
    def calculate_polygon_elevation_stats(self, polygon_points: List[GeoPoint]) -> Dict:
        """Calculate elevation statistics for polygon vertices"""
        elevations = [p.elevation for p in polygon_points if p.elevation is not None]
        
        if not elevations:
            return {"status": "No elevation data available"}
        
        return {
            "min_elevation": min(elevations),
            "max_elevation": max(elevations),
            "avg_elevation": sum(elevations) / len(elevations),
            "elevation_range": max(elevations) - min(elevations),
            "vertex_elevations": elevations
        }
    
    def analyze_image(self, image_path: str, 
                     polygon_geo_points: List[GeoPoint],
                     image_bounds: Tuple[float, float, float, float],
                     pixel_to_meter_ratio: float = 0.1,
                     use_elevation_api: str = "open") -> Dict:
        """
        Main analysis function with geographic coordinates
        
        Args:
            image_path: Path to satellite image
            polygon_geo_points: List of GeoPoint objects defining the polygon
            image_bounds: (north_lat, south_lat, east_lng, west_lng) of the image
            pixel_to_meter_ratio: Scaling factor for height estimation
            use_elevation_api: "google", "open", or "none"
        """
        results = {
            'objects': [],
            'polygon_info': {},
            'depth_map': None,
            'processed_image': None,
            'polygon_geo_points': polygon_geo_points,
            'image_bounds': image_bounds
        }
        
        # Load image
        image = self.load_image(image_path)
        print(f"Image loaded: {image.shape}")
        
        # Setup georeferencing
        self.setup_georeferencing(image_bounds, image.shape[:2])
        
        # Get elevation data for polygon vertices
        if use_elevation_api == "google":
            polygon_geo_points = self.get_elevation_google(polygon_geo_points)
        elif use_elevation_api == "open":
            polygon_geo_points = self.get_elevation_open(polygon_geo_points)
        
        # Calculate polygon elevation statistics
        results['polygon_info'] = self.calculate_polygon_elevation_stats(polygon_geo_points)
        
        # Process polygon area
        image, mask, polygon_pixels = self.process_polygon_area(image, polygon_geo_points)
        print("✓ Polygon area processed")
        
        # Detect objects
        objects = self.detect_objects_contour(image)
        print(f"✓ Found {len(objects)} objects")
        
        # Estimate depth map
        try:
            depth_map = self.estimate_depth_midas(image)
            results['depth_map'] = depth_map
            print("✓ Depth map generated")
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            depth_map = None
        
        # Analyze each object and get geographic coordinates
        object_geo_points = []
        detected_objects = []
        
        for i, obj in enumerate(objects):
            bbox = obj['bbox']
            x, y, w, h = bbox
            
            # Calculate object center in geographic coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            center_geo = self.pixel_to_geo(center_x, center_y)
            object_geo_points.append(center_geo)
            
            # Calculate height estimates
            height_estimates = {}
            
            # Method 1: Depth-based height estimation
            if depth_map is not None:
                height_depth = self.estimate_height_from_depth(
                    depth_map, bbox, pixel_to_meter_ratio=pixel_to_meter_ratio
                )
                height_estimates['depth_based'] = height_depth
            
            # Method 2: Pixel height (needs calibration)
            pixel_height = h * pixel_to_meter_ratio
            height_estimates['pixel_based'] = pixel_height
            
            # Create detected object
            detected_obj = DetectedObject(
                id=i,
                bbox_pixels=bbox,
                center_geo=center_geo,
                area_pixels=obj['area'],
                height_estimates=height_estimates,
                confidence=0.7  # Placeholder confidence score
            )
            
            detected_objects.append(detected_obj)
        
        # Get elevation data for detected objects
        if object_geo_points:
            if use_elevation_api == "google":
                object_geo_points = self.get_elevation_google(object_geo_points)
            elif use_elevation_api == "open":
                object_geo_points = self.get_elevation_open(object_geo_points)
            
            # Update detected objects with elevation data
            for i, obj in enumerate(detected_objects):
                if i < len(object_geo_points):
                    obj.elevation = object_geo_points[i].elevation
                    obj.center_geo.elevation = object_geo_points[i].elevation
        
        results['objects'] = detected_objects
        
        # Create visualization
        results['processed_image'] = self.visualize_results(
            image, detected_objects, depth_map, polygon_pixels
        )
        
        return results
    
    def visualize_results(self, image, detected_objects: List[DetectedObject], 
                         depth_map=None, polygon_pixels=None):
        """Create visualization of detected objects and heights"""
        vis_image = image.copy()
        
        # Draw polygon
        if polygon_pixels:
            polygon_array = np.array(polygon_pixels, dtype=np.int32)
            cv2.polylines(vis_image, [polygon_array], True, (255, 0, 0), 3)
        
        # Draw bounding boxes and information
        for obj in detected_objects:
            x, y, w, h = obj.bbox_pixels
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare label with geographic info
            height_est = obj.height_estimates.get('depth_based', 
                                                obj.height_estimates.get('pixel_based', 0))
            
            label_lines = [
                f"ID: {obj.id}",
                f"H: {height_est:.1f}m",
                f"Lat: {obj.center_geo.lat:.6f}",
                f"Lng: {obj.center_geo.lng:.6f}"
            ]
            
            if obj.elevation is not None:
                label_lines.append(f"Elev: {obj.elevation:.1f}m")
            
            # Draw multi-line label
            for i, line in enumerate(label_lines):
                y_offset = y - 10 - (len(label_lines) - 1 - i) * 15
                cv2.putText(vis_image, line, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis_image
    
    def save_results(self, results: Dict, output_path: str):
        """Save analysis results and visualizations"""
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original with detections
        if results['processed_image'] is not None:
            axes[0, 0].imshow(cv2.cvtColor(results['processed_image'], cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"Detected Objects ({len(results['objects'])})")
            axes[0, 0].axis('off')
        
        # Depth map
        if results['depth_map'] is not None:
            im = axes[0, 1].imshow(results['depth_map'], cmap='plasma')
            axes[0, 1].set_title("Depth Map")
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Height analysis
        if results['objects']:
            heights = []
            elevations = []
            for obj in results['objects']:
                height = obj.height_estimates.get('depth_based', 
                                                obj.height_estimates.get('pixel_based', 0))
                heights.append(height)
                elevations.append(obj.elevation if obj.elevation else 0)
            
            axes[1, 0].bar(range(len(heights)), heights)
            axes[1, 0].set_title("Estimated Object Heights")
            axes[1, 0].set_xlabel("Object ID")
            axes[1, 0].set_ylabel("Height (m)")
            
            axes[1, 1].bar(range(len(elevations)), elevations)
            axes[1, 1].set_title("Object Ground Elevations")
            axes[1, 1].set_xlabel("Object ID")
            axes[1, 1].set_ylabel("Elevation (m)")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed summary
        self.print_detailed_summary(results)
        
        # Save JSON report
        self.save_json_report(results, output_path.replace('.png', '_report.json'))
    
    def print_detailed_summary(self, results: Dict):
        """Print detailed analysis summary"""
        print("\n" + "="*60)
        print("GEOGRAPHIC OBJECT ANALYSIS SUMMARY")
        print("="*60)
        
        # Polygon information
        print(f"\nPOLYGON ANALYSIS:")
        polygon_info = results['polygon_info']
        for key, value in polygon_info.items():
            if key != 'vertex_elevations':
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.2f}m")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Object details
        print(f"\nDETECTED OBJECTS: {len(results['objects'])}")
        print("-" * 40)
        
        for obj in results['objects']:
            print(f"\nObject {obj.id}:")
            print(f"  Location: {obj.center_geo.lat:.6f}°N, {obj.center_geo.lng:.6f}°E")
            if obj.elevation is not None:
                print(f"  Ground Elevation: {obj.elevation:.2f}m")
            print(f"  Bounding Box (pixels): {obj.bbox_pixels}")
            print(f"  Area: {obj.area_pixels} pixels")
            print(f"  Height Estimates:")
            for method, height in obj.height_estimates.items():
                print(f"    {method.replace('_', ' ').title()}: {height:.2f}m")
            if obj.elevation is not None and 'depth_based' in obj.height_estimates:
                total_height = obj.elevation + obj.height_estimates['depth_based']
                print(f"  Total Height Above Sea Level: {total_height:.2f}m")
            print(f"  Confidence: {obj.confidence:.2f}")
        
        # Image bounds
        print(f"\nIMAGE GEOGRAPHIC BOUNDS:")
        bounds = results['image_bounds']
        print(f"  North: {bounds[0]:.6f}°")
        print(f"  South: {bounds[1]:.6f}°")
        print(f"  East: {bounds[2]:.6f}°")
        print(f"  West: {bounds[3]:.6f}°")
    
    def save_json_report(self, results: Dict, json_path: str):
        """Save detailed results to JSON file"""
        # Convert results to JSON-serializable format
        json_results = {
            'image_bounds': results['image_bounds'],
            'polygon_info': results['polygon_info'],
            'polygon_vertices': [
                {
                    'lat': p.lat,
                    'lng': p.lng,
                    'elevation': p.elevation
                } for p in results['polygon_geo_points']
            ],
            'objects': [
                {
                    'id': obj.id,
                    'location': {
                        'lat': obj.center_geo.lat,
                        'lng': obj.center_geo.lng,
                        'elevation': obj.elevation
                    },
                    'bbox_pixels': obj.bbox_pixels,
                    'area_pixels': obj.area_pixels,
                    'height_estimates': obj.height_estimates,
                    'confidence': obj.confidence
                } for obj in results['objects']
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Detailed report saved to: {json_path}")

# Example usage
def main():
    """Example usage with geographic coordinates"""
    
    # Initialize detector (optionally with Google API key for better elevation data)
    detector = GeoObjectHeightDetector(elevation_api_key=None)  # Add your API key here
    
    # Define polygon vertices using lat/lng coordinates
    polygon_vertices = [
        GeoPoint(lat=37.7749, lng=-122.4194),  # San Francisco example
        GeoPoint(lat=37.7849, lng=-122.4194),
        GeoPoint(lat=37.7849, lng=-122.4094),
        GeoPoint(lat=37.7749, lng=-122.4094)
    ]
    
    # Define image geographic bounds (north_lat, south_lat, east_lng, west_lng)
    image_bounds = (37.7900, 37.7700, -122.4000, -122.4300)
    
    # Analyze image
    results = detector.analyze_image(
        image_path="path/to/your/satellite_image.jpg",  # Replace with your image
        polygon_geo_points=polygon_vertices,
        image_bounds=image_bounds,
        pixel_to_meter_ratio=0.15,  # Adjust based on image resolution
        use_elevation_api="open"  # "google", "open", or "none"
    )
    
    # Save results
    detector.save_results(results, "geo_height_analysis_results.png")
    
    return results

if __name__ == "__main__":
    print("Geographic Object Height Detection using AI")
    print("="*50)
    print("Features:")
    print("• Input: Polygon with lat/lng vertices")
    print("• Output: Object locations, elevations, and heights")
    print("• Elevation data from Open Elevation API (free)")
    print("• Geographic coordinate conversion")
    print("\nReplace image path and coordinates with your data")
    
    # Uncomment to run with actual data:
    # results = main()
