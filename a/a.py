import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import math
from typing import List, Tuple, Dict, Optional
import json
import os
import requests
from urllib.parse import urlencode
import base64
from io import BytesIO
from PIL import Image

class GoogleMapsVisionDetector:
    def __init__(self, google_api_key: str, yolo_config_path: str = None, 
                 yolo_weights_path: str = None, yolo_classes_path: str = None):
        """
        Initialize the vision obstruction detector with Google Maps/Street View and YOLO
        
        Args:
            google_api_key: Google Cloud API key with Maps and Street View enabled
            yolo_config_path: Path to YOLO config file (.cfg)
            yolo_weights_path: Path to YOLO weights file (.weights)
            yolo_classes_path: Path to YOLO class names file (.names)
        """
        self.google_api_key = google_api_key
        
        # Google APIs endpoints
        self.streetview_api_url = "https://maps.googleapis.com/maps/api/streetview"
        self.geocoding_api_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.directions_api_url = "https://maps.googleapis.com/maps/api/directions/json"
        
        # Common street obstacles that can obstruct vision
        self.obstruction_classes = {
            'car': {'min_height': 1.2, 'blocks_vision': True},
            'truck': {'min_height': 2.0, 'blocks_vision': True},
            'bus': {'min_height': 2.5, 'blocks_vision': True},
            'motorcycle': {'min_height': 1.0, 'blocks_vision': True},
            'bicycle': {'min_height': 1.0, 'blocks_vision': False},
            'person': {'min_height': 1.7, 'blocks_vision': True},
            'traffic light': {'min_height': 1.5, 'blocks_vision': True},
            'stop sign': {'min_height': 1.0, 'blocks_vision': True},
            'fire hydrant': {'min_height': 0.8, 'blocks_vision': True}
        }
        
        # Initialize YOLO
        self.yolo_net = None
        self.yolo_classes = []
        self.output_layers = []
        
        if yolo_config_path and yolo_weights_path and yolo_classes_path:
            self.load_yolo_model(yolo_config_path, yolo_weights_path, yolo_classes_path)
        else:
            print("YOLO model paths not provided. Using default COCO classes.")
            self.setup_default_classes()
    
    def setup_default_classes(self):
        """Setup default COCO class names for YOLO"""
        self.yolo_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    
    def load_yolo_model(self, config_path: str, weights_path: str, classes_path: str):
        """Load YOLO model from files"""
        try:
            self.yolo_net = cv2.dnn.readNet(weights_path, config_path)
            
            with open(classes_path, 'r') as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            
            layer_names = self.yolo_net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            
            print(f"YOLO model loaded successfully with {len(self.yolo_classes)} classes")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_net = None
    
    def get_street_view_image(self, lat: float, lng: float, heading: int = 0, 
                            pitch: int = 0, fov: int = 90, size: str = "640x640") -> Optional[np.ndarray]:
        """
        Fetch Street View image from Google Street View Static API
        
        Args:
            lat: Latitude
            lng: Longitude  
            heading: Camera heading (0-360 degrees, 0=North, 90=East, 180=South, 270=West)
            pitch: Camera pitch (-90 to 90 degrees, 0=horizontal)
            fov: Field of view (10-120 degrees)
            size: Image size (e.g., "640x640", "400x400")
            
        Returns:
            OpenCV image array or None if failed
        """
        params = {
            'location': f"{lat},{lng}",
            'heading': heading,
            'pitch': pitch,
            'fov': fov,
            'size': size,
            'key': self.google_api_key
        }
        
        url = f"{self.streetview_api_url}?{urlencode(params)}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to OpenCV format
            image = Image.open(BytesIO(response.content))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            print(f"Street View image downloaded: {lat:.6f}, {lng:.6f}, heading={heading}°")
            return cv_image
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Street View image: {e}")
            return None
    
    def get_street_view_panorama(self, lat: float, lng: float, output_dir: str = "streetview_images") -> List[Dict]:
        """
        Get panoramic Street View images (360-degree coverage) from a location
        
        Args:
            lat: Latitude
            lng: Longitude
            output_dir: Directory to save images
            
        Returns:
            List of image info with headings and file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        images_info = []
        
        # Capture images every 45 degrees for full coverage
        headings = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for heading in headings:
            image = self.get_street_view_image(lat, lng, heading=heading, fov=90)
            
            if image is not None:
                # Save image
                filename = f"streetview_{lat:.6f}_{lng:.6f}_h{heading:03d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, image)
                
                images_info.append({
                    'heading': heading,
                    'filepath': filepath,
                    'lat': lat,
                    'lng': lng,
                    'image_array': image
                })
        
        print(f"Downloaded {len(images_info)} panoramic images for location {lat:.6f}, {lng:.6f}")
        return images_info
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to GPS coordinates using Google Geocoding API
        
        Args:
            address: Street address
            
        Returns:
            (lat, lng) tuple or None if failed
        """
        params = {
            'address': address,
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(self.geocoding_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return (location['lat'], location['lng'])
            else:
                print(f"Geocoding failed: {data.get('status', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error in geocoding: {e}")
            return None
    
    def calculate_gps_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between GPS points using Haversine formula"""
        R = 6378137.0  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def gps_to_local_coordinates(self, lat: float, lng: float, 
                               ref_lat: float, ref_lng: float) -> Tuple[float, float]:
        """Convert GPS to local Cartesian coordinates"""
        R = 6378137.0
        ref_lat_rad = math.radians(ref_lat)
        
        x = R * math.radians(lng - ref_lng) * math.cos(ref_lat_rad)
        y = R * math.radians(lat - ref_lat)
        
        return (x, y)
    
    def create_search_polygon_gps(self, center_lat: float, center_lng: float,
                                polygon_type: str = 'rectangle', 
                                width_meters: float = 20.0, 
                                height_meters: float = 10.0,
                                custom_gps_points: List[Tuple[float, float]] = None) -> Tuple[Polygon, List[Tuple[float, float]]]:
        """Create polygon search area around GPS coordinates"""
        
        if polygon_type == 'custom' and custom_gps_points:
            # Use custom GPS points directly
            local_points = []
            for lat, lng in custom_gps_points:
                x, y = self.gps_to_local_coordinates(lat, lng, center_lat, center_lng)
                local_points.append((x, y))
            gps_points = custom_gps_points
        else:
            # Create standard shapes in local coordinates then convert to GPS
            if polygon_type == 'rectangle':
                local_points = [
                    (-width_meters/2, -height_meters/2),
                    (width_meters/2, -height_meters/2),
                    (width_meters/2, height_meters/2),
                    (-width_meters/2, height_meters/2)
                ]
            elif polygon_type == 'triangle':
                local_points = [
                    (0, 0),
                    (-width_meters/2, height_meters),
                    (width_meters/2, height_meters)
                ]
            elif polygon_type == 'sector':
                # 90-degree sector
                radius = width_meters
                local_points = [(0, 0)]
                for i in range(21):  # 21 points for smooth arc
                    angle = math.radians(-45 + i * 4.5)  # -45° to +45°
                    x = radius * math.sin(angle)
                    y = radius * math.cos(angle)
                    local_points.append((x, y))
            else:  # Default rectangle
                local_points = [
                    (-width_meters/2, -height_meters/2),
                    (width_meters/2, -height_meters/2),
                    (width_meters/2, height_meters/2),
                    (-width_meters/2, height_meters/2)
                ]
            
            # Convert local points to GPS
            gps_points = []
            for x, y in local_points:
                lat = center_lat + math.degrees(y / 6378137.0)
                lng = center_lng + math.degrees(x / (6378137.0 * math.cos(math.radians(center_lat))))
                gps_points.append((lat, lng))
        
        polygon = Polygon(local_points)
        return polygon, gps_points
    
    def detect_objects_yolo(self, image: np.ndarray, confidence_threshold: float = 0.5, 
                           nms_threshold: float = 0.4) -> List[Dict]:
        """Detect objects using YOLO model on image array"""
        if image is None:
            return []
        
        height, width = image.shape[:2]
        
        if self.yolo_net is None:
            return self._mock_detections(width, height)
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo_net.setInput(blob)
        outputs = self.yolo_net.forward(self.output_layers)
        
        # Process YOLO outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.yolo_classes[class_ids[i]]
                confidence = confidences[i]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [x + w//2, y + h//2]
                })
        
        return detections
    
    def _mock_detections(self, width: int, height: int) -> List[Dict]:
        """Generate mock detections for testing"""
        return [
            {
                'class': 'car',
                'confidence': 0.85,
                'bbox': [int(width*0.1), int(height*0.6), int(width*0.2), int(height*0.3)],
                'center': [int(width*0.2), int(height*0.75)]
            },
            {
                'class': 'truck',
                'confidence': 0.92,
                'bbox': [int(width*0.4), int(height*0.5), int(width*0.25), int(height*0.4)],
                'center': [int(width*0.525), int(height*0.7)]
            },
            {
                'class': 'person',
                'confidence': 0.78,
                'bbox': [int(width*0.7), int(height*0.4), int(width*0.08), int(height*0.5)],
                'center': [int(width*0.74), int(height*0.65)]
            }
        ]
    
    def analyze_driveway_visibility(self, driveway_location: str, 
                                  polygon_type: str = 'sector',
                                  search_radius_meters: float = 15.0,
                                  vision_headings: List[int] = None) -> Dict:
        """
        Complete driveway visibility analysis using Google Street View
        
        Args:
            driveway_location: Address or "lat,lng" string
            polygon_type: Search area shape
            search_radius_meters: Search area size
            vision_headings: Camera angles to analyze (default: [0, 90, 180, 270])
            
        Returns:
            Complete analysis results with Street View images and detections
        """
        # Parse location
        if ',' in driveway_location and len(driveway_location.split(',')) == 2:
            # GPS coordinates provided
            try:
                lat, lng = map(float, driveway_location.split(','))
            except ValueError:
                print("Invalid GPS coordinates format. Use 'lat,lng'")
                return {}
        else:
            # Address provided - geocode it
            coords = self.geocode_address(driveway_location)
            if coords is None:
                print(f"Could not geocode address: {driveway_location}")
                return {}
            lat, lng = coords
        
        print(f"Analyzing driveway visibility at: {lat:.6f}, {lng:.6f}")
        
        # Default vision headings (cardinal directions)
        if vision_headings is None:
            vision_headings = [0, 90, 180, 270]  # North, East, South, West
        
        # Create search polygon
        search_polygon, gps_polygon_points = self.create_search_polygon_gps(
            lat, lng, polygon_type, search_radius_meters, search_radius_meters
        )
        
        all_obstructions = []
        streetview_images = []
        
        # Analyze each viewing direction
        for heading in vision_headings:
            print(f"Analyzing heading {heading}° from driveway...")
            
            # Get Street View image
            image = self.get_street_view_image(lat, lng, heading=heading, fov=90)
            
            if image is None:
                print(f"Could not get Street View image for heading {heading}°")
                continue
            
            # Save image for reference
            image_filename = f"driveway_view_{lat:.6f}_{lng:.6f}_h{heading:03d}.jpg"
            cv2.imwrite(image_filename, image)
            
            streetview_images.append({
                'heading': heading,
                'filename': image_filename,
                'image_array': image
            })
            
            # Detect objects in this view
            detections = self.detect_objects_yolo(image)
            
            # Estimate object locations (simplified - assumes objects are at road level)
            for detection in detections:
                obj_class = detection['class']
                if obj_class in self.obstruction_classes:
                    if self.obstruction_classes[obj_class]['blocks_vision']:
                        # Estimate GPS location based on pixel position and heading
                        # This is simplified - real implementation would need depth estimation
                        estimated_distance = 10.0  # Assume 10m distance
                        bearing_offset = (detection['center'][0] - 320) / 320.0 * 45  # ±45° FOV
                        object_bearing = (heading + bearing_offset) % 360
                        
                        # Calculate estimated object GPS coordinates
                        bearing_rad = math.radians(object_bearing)
                        obj_lat = lat + math.degrees(estimated_distance * math.cos(bearing_rad) / 111111.0)
                        obj_lng = lng + math.degrees(estimated_distance * math.sin(bearing_rad) / (111111.0 * math.cos(math.radians(lat))))
                        
                        # Check if estimated location is within search polygon
                        local_x, local_y = self.gps_to_local_coordinates(obj_lat, obj_lng, lat, lng)
                        if search_polygon.contains(Point(local_x, local_y)):
                            distance = self.calculate_gps_distance(lat, lng, obj_lat, obj_lng)
                            
                            all_obstructions.append({
                                'class': obj_class,
                                'confidence': detection['confidence'],
                                'estimated_gps': (obj_lat, obj_lng),
                                'distance_meters': distance,
                                'viewing_heading': heading,
                                'bbox': detection['bbox'],
                                'estimated_location': True  # Flag indicating this is estimated
                            })
        
        # Calculate visibility metrics
        total_obstructions = len(all_obstructions)
        critical_obstructions = [obs for obs in all_obstructions if obs['distance_meters'] < 5.0]
        vehicle_obstructions = [obs for obs in all_obstructions if obs['class'] in ['car', 'truck', 'bus', 'motorcycle']]
        
        # Vision quality score (0-100)
        vision_score = max(0, 100 - (total_obstructions * 10) - (len(critical_obstructions) * 20))
        
        # Generate recommendations
        recommendations = []
        if total_obstructions == 0:
            recommendations.append("Excellent visibility - no obstructions detected")
        else:
            if critical_obstructions:
                recommendations.append(f"CAUTION: {len(critical_obstructions)} critical obstructions within 5m")
            if vehicle_obstructions:
                recommendations.append(f"Vehicle obstructions detected - check for parked cars")
            if vision_score < 50:
                recommendations.append("Poor visibility - consider installing safety mirrors or warning systems")
        
        results = {
            'location': {
                'address': driveway_location,
                'gps_coordinates': (lat, lng)
            },
            'analysis_parameters': {
                'polygon_type': polygon_type,
                'search_radius_meters': search_radius_meters,
                'vision_headings_analyzed': vision_headings,
                'polygon_gps_points': gps_polygon_points
            },
            'street_view_images': streetview_images,
            'detection_results': {
                'total_obstructions': total_obstructions,
                'critical_obstructions': len(critical_obstructions),
                'vehicle_obstructions': len(vehicle_obstructions),
                'vision_quality_score': vision_score,
                'obstructions': all_obstructions
            },
            'safety_recommendations': recommendations
        }
        
        return results
    
    def create_visibility_report(self, results: Dict, output_file: str = "visibility_report.html"):
        """
        Generate an HTML report with Street View images and analysis results
        
        Args:
            results: Analysis results from analyze_driveway_visibility
            output_file: Output HTML file path
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Driveway Visibility Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                .image-container {{ text-align: center; }}
                .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .danger {{ color: red; }}
                .obstruction {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Driveway Visibility Analysis Report</h1>
                <p><strong>Location:</strong> {results['location']['address']}</p>
                <p><strong>GPS:</strong> {results['location']['gps_coordinates'][0]:.6f}, {results['location']['gps_coordinates'][1]:.6f}</p>
                <p><strong>Analysis Date:</strong> {json.dumps(results['analysis_parameters'], indent=2)}</p>
            </div>
            
            <div class="section">
                <h2>Visibility Score</h2>
                <div class="score {'good' if results['detection_results']['vision_quality_score'] >= 70 else 'warning' if results['detection_results']['vision_quality_score'] >= 40 else 'danger'}">
                    {results['detection_results']['vision_quality_score']}/100
                </div>
            </div>
            
            <div class="section">
                <h2>Street View Analysis</h2>
                <div class="image-grid">
        """
        
        for img_info in results['street_view_images']:
            html_content += f"""
                    <div class="image-container">
                        <img src="{img_info['filename']}" alt="View at {img_info['heading']}°">
                        <p>Heading: {img_info['heading']}°</p>
                    </div>
            """
        
        html_content += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>Detected Obstructions</h2>
                <p><strong>Total:</strong> {results['detection_results']['total_obstructions']}</p>
                <p><strong>Critical (within 5m):</strong> {results['detection_results']['critical_obstructions']}</p>
                <p><strong>Vehicles:</strong> {results['detection_results']['vehicle_obstructions']}</p>
        """
        
        for obs in results['detection_results']['obstructions']:
            html_content += f"""
                <div class="obstruction">
                    <strong>{obs['class'].title()}</strong> - 
                    Distance: {obs['distance_meters']:.1f}m, 
                    Confidence: {obs['confidence']:.0%}, 
                    View: {obs['viewing_heading']}°
                </div>
            """
        
        html_content += f"""
            </div>
            
            <div class="section">
                <h2>Safety Recommendations</h2>
                <ul>
        """
        
        for rec in results['safety_recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Visibility report saved to: {output_file}")

# Example usage
def main():
    # Replace with your Google Cloud API key
    api_key = "YOUR_GOOGLE_CLOUD_API_KEY"
    
    if api_key == "YOUR_GOOGLE_CLOUD_API_KEY":
        print("Please set your Google Cloud API key!")
        print("Get one at: https://console.cloud.google.com/apis/credentials")
        print("Enable: Maps JavaScript API, Street View Static API, Geocoding API")
        return
    
    # Initialize detector
    detector = GoogleMapsVisionDetector(api_key)
    
    print("Google Maps + Street View Vision Obstruction Analysis")
    print("=" * 55)
    
    # Example analysis - you can use either address or GPS coordinates
    locations_to_test = [
        "1600 Amphitheatre Parkway, Mountain View, CA",  # Google HQ
        "Times Square, New York, NY",
        "37.7749,-122.4194"  # San Francisco GPS coordinates
    ]
    
    for location in locations_to_test:
        print(f"\n--- Analyzing: {location} ---")
        
        results = detector.analyze_driveway_visibility(
            driveway_location=location,
            polygon_type='sector',
            search_radius_meters=20.0,
            vision_headings=[0, 90, 180, 270]  # Check all directions
        )
        
        if results:
            print(f"Vision Score: {results['detection_results']['vision_quality_score']}/100")
            print(f"Total Obstructions: {results['detection_results']['total_obstructions']}")
            print(f"Critical Obstructions: {results['detection_results']['critical_obstructions']}")
            print("Recommendations:")
            for rec in results['safety_recommendations']:
                print(f"  • {rec}")
            
            # Generate report
            report_filename = f"report_{location.replace(' ', '_').replace(',', '')}.html"
            detector.create_visibility_report(results, report_filename)
        else:
            print("Analysis failed for this location")

if __name__ == "__main__":
    main()
