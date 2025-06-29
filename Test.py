import os
import cv2
import numpy as np
import requests
import json
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import google.generativeai as genai
from google.cloud import vision
import folium
from geopy.distance import distance
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Object detection result with coordinates"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pixel_center: Tuple[int, int]
    gps_coordinates: Optional[Tuple[float, float]] = None
    estimated_distance: Optional[float] = None

@dataclass
class CameraParams:
    """Street View camera parameters"""
    lat: float
    lng: float
    heading: float
    pitch: float
    fov: float = 90
    image_width: int = 640
    image_height: int = 640

class GoogleMapsAPI:
    """Google Maps API integration for Street View and Geocoding"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
    
    def get_street_view_image(self, lat: float, lng: float, heading: float = 0, 
                            pitch: float = 0, size: str = "640x640") -> np.ndarray:
        """Fetch Street View image and return as numpy array"""
        url = f"{self.base_url}/streetview"
        params = {
            'size': size,
            'location': f"{lat},{lng}",
            'heading': heading,
            'pitch': pitch,
            'key': self.api_key,
            'fov': 90
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(requests.get(url, params=params, stream=True).raw)
            return np.array(image)
        except Exception as e:
            logger.error(f"Error fetching Street View image: {e}")
            return None
    
    def get_street_view_metadata(self, lat: float, lng: float) -> Dict:
        """Get Street View metadata for coordinate mapping"""
        url = f"{self.base_url}/streetview/metadata"
        params = {
            'location': f"{lat},{lng}",
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return {}
    
    def geocode_address(self, address: str) -> Tuple[float, float]:
        """Convert address to coordinates"""
        url = f"{self.base_url}/geocode/json"
        params = {
            'address': address,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                location = data['results'][0]['geometry']['location']
                return location['lat'], location['lng']
        except Exception as e:
            logger.error(f"Error geocoding address: {e}")
        
        return None, None

class ObjectDetector:
    """AI-based object detection using multiple models"""
    
    def __init__(self):
        self.yolo_model = None
        self.vision_client = None
        self.custom_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Object class mappings for different detection types
        self.infrastructure_classes = {
            'fence': ['fence', 'barrier', 'railing'],
            'driveway': ['driveway', 'parking', 'pavement'],
            'street': ['road', 'street', 'highway'],
            'mailbox': ['mailbox', 'post box'],
            'fire_hydrant': ['fire hydrant', 'hydrant'],
            'stop_sign': ['stop sign'],
            'traffic_light': ['traffic light'],
            'building': ['building', 'house'],
            'tree': ['tree'],
            'car': ['car', 'vehicle', 'truck']
        }
    
    def load_yolo_model(self, model_path: str = 'yolov8n.pt'):
        """Load YOLO model for general object detection"""
        try:
            self.yolo_model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
    
    def load_google_vision_client(self, credentials_path: str):
        """Load Google Cloud Vision client"""
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.vision_client = vision.ImageAnnotatorClient()
            logger.info("Google Vision client loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Google Vision client: {e}")
    
    def detect_with_yolo(self, image: np.ndarray) -> List[Detection]:
        """Detect objects using YOLO model"""
        if self.yolo_model is None:
            return []
        
        detections = []
        results = self.yolo_model(image)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Filter for infrastructure objects
                    if self._is_infrastructure_object(class_name) and confidence > 0.5:
                        pixel_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        
                        detection = Detection(
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            pixel_center=pixel_center
                        )
                        detections.append(detection)
        
        return detections
    
    def detect_with_google_vision(self, image: np.ndarray) -> List[Detection]:
        """Detect objects using Google Cloud Vision"""
        if self.vision_client is None:
            return []
        
        detections = []
        
        # Convert numpy array to bytes
        _, encoded_image = cv2.imencode('.jpg', image)
        image_bytes = encoded_image.tobytes()
        
        try:
            # Object localization
            image_vision = vision.Image(content=image_bytes)
            objects = self.vision_client.object_localization(image=image_vision).localized_object_annotations
            
            for obj in objects:
                if self._is_infrastructure_object(obj.name) and obj.score > 0.5:
                    # Get bounding box
                    vertices = obj.bounding_poly.normalized_vertices
                    h, w = image.shape[:2]
                    
                    x1 = int(vertices[0].x * w)
                    y1 = int(vertices[0].y * h)
                    x2 = int(vertices[2].x * w)
                    y2 = int(vertices[2].y * h)
                    
                    pixel_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    detection = Detection(
                        class_name=obj.name.lower(),
                        confidence=obj.score,
                        bbox=(x1, y1, x2, y2),
                        pixel_center=pixel_center
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error in Google Vision detection: {e}")
        
        return detections
    
    def detect_custom_infrastructure(self, image: np.ndarray) -> List[Detection]:
        """Custom detection for specific infrastructure objects"""
        detections = []
        
        # Fence detection using edge detection and contours
        fence_detections = self._detect_fences(image)
        detections.extend(fence_detections)
        
        # Driveway detection using color/texture analysis
        driveway_detections = self._detect_driveways(image)
        detections.extend(driveway_detections)
        
        return detections
    
    def _detect_fences(self, image: np.ndarray) -> List[Detection]:
        """Custom fence detection using computer vision techniques"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for linear structures
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter contours that might be fences (linear, certain size)
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Reasonable fence size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio for fence-like structures
                aspect_ratio = w / h if h > 0 else 0
                if 2 < aspect_ratio < 10 or 0.1 < aspect_ratio < 0.5:
                    pixel_center = (x + w // 2, y + h // 2)
                    
                    detection = Detection(
                        class_name='fence',
                        confidence=0.7,  # Custom confidence
                        bbox=(x, y, x + w, y + h),
                        pixel_center=pixel_center
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_driveways(self, image: np.ndarray) -> List[Detection]:
        """Custom driveway detection using color and texture analysis"""
        detections = []
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common driveway materials
        # Concrete/asphalt ranges
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 30, 200])
        
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 20000:  # Reasonable driveway size
                x, y, w, h = cv2.boundingRect(contour)
                pixel_center = (x + w // 2, y + h // 2)
                
                detection = Detection(
                    class_name='driveway',
                    confidence=0.6,
                    bbox=(x, y, x + w, y + h),
                    pixel_center=pixel_center
                )
                detections.append(detection)
        
        return detections
    
    def _is_infrastructure_object(self, class_name: str) -> bool:
        """Check if detected object is relevant infrastructure"""
        class_name_lower = class_name.lower()
        for category, names in self.infrastructure_classes.items():
            if any(name in class_name_lower for name in names):
                return True
        return False

class CoordinateMapper:
    """Convert pixel coordinates to GPS coordinates"""
    
    @staticmethod
    def pixel_to_gps(detection: Detection, camera_params: CameraParams) -> Tuple[float, float]:
        """Convert pixel coordinates to GPS coordinates"""
        # Extract pixel coordinates
        pixel_x, pixel_y = detection.pixel_center
        
        # Image dimensions
        img_width = camera_params.image_width
        img_height = camera_params.image_height
        
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        norm_x = (pixel_x - img_width / 2) / (img_width / 2)
        norm_y = (pixel_y - img_height / 2) / (img_height / 2)
        
        # Calculate angular offset from camera center
        fov_rad = math.radians(camera_params.fov)
        horizontal_angle = norm_x * (fov_rad / 2)
        vertical_angle = norm_y * (fov_rad / 2)
        
        # Calculate bearing from camera heading
        bearing = camera_params.heading + math.degrees(horizontal_angle)
        bearing = bearing % 360  # Normalize to 0-360
        
        # Estimate distance based on object size and type
        estimated_distance = CoordinateMapper.estimate_distance(detection, camera_params)
        
        # Calculate GPS coordinates using bearing and distance
        gps_lat, gps_lng = CoordinateMapper.offset_coordinate(
            camera_params.lat, camera_params.lng, estimated_distance, bearing
        )
        
        return gps_lat, gps_lng
    
    @staticmethod
    def estimate_distance(detection: Detection, camera_params: CameraParams) -> float:
        """Estimate distance to object based on size and type"""
        # Object size in pixels
        bbox = detection.bbox
        pixel_height = bbox[3] - bbox[1]
        pixel_width = bbox[2] - bbox[0]
        
        # Known approximate real-world sizes (in meters)
        object_sizes = {
            'mailbox': 1.2,  # Height
            'fire_hydrant': 1.0,
            'stop_sign': 2.4,
            'car': 1.8,
            'tree': 5.0,
            'fence': 1.5,
            'building': 8.0
        }
        
        # Get expected real-world size
        real_size = object_sizes.get(detection.class_name, 2.0)  # Default 2m
        
        # Simple distance estimation using perspective
        # This is a rough approximation - more sophisticated methods could be used
        if pixel_height > 0:
            # Distance = (real_size * focal_length) / pixel_size
            # Assuming focal_length approximation based on FOV
            focal_length = camera_params.image_height / (2 * math.tan(math.radians(camera_params.fov / 2)))
            distance_estimate = (real_size * focal_length) / pixel_height
            
            # Clamp distance to reasonable values
            return max(5.0, min(100.0, distance_estimate))
        
        return 20.0  # Default distance
    
    @staticmethod
    def offset_coordinate(lat: float, lng: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
        """Calculate new coordinates given distance and bearing"""
        # Earth radius in meters
        R = 6378137.0
        
        # Convert to radians
        bearing_rad = math.radians(bearing_deg)
        lat_rad = math.radians(lat)
        lng_rad = math.radians(lng)
        
        # Calculate new latitude
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_m / R) +
            math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad)
        )
        
        # Calculate new longitude
        new_lng_rad = lng_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
            math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Convert back to degrees
        new_lat = math.degrees(new_lat_rad)
        new_lng = math.degrees(new_lng_rad)
        
        return new_lat, new_lng

class DataManager:
    """Manage detection data storage and retrieval"""
    
    def __init__(self, db_path: str = 'object_detections.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                camera_lat REAL,
                camera_lng REAL,
                camera_heading REAL,
                class_name TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                pixel_x INTEGER,
                pixel_y INTEGER,
                gps_lat REAL,
                gps_lng REAL,
                estimated_distance REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_detections(self, detections: List[Detection], camera_params: CameraParams):
        """Save detections to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for detection in detections:
            cursor.execute('''
                INSERT INTO detections (
                    timestamp, camera_lat, camera_lng, camera_heading,
                    class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    pixel_x, pixel_y, gps_lat, gps_lng, estimated_distance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, camera_params.lat, camera_params.lng, camera_params.heading,
                detection.class_name, detection.confidence,
                detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3],
                detection.pixel_center[0], detection.pixel_center[1],
                detection.gps_coordinates[0] if detection.gps_coordinates else None,
                detection.gps_coordinates[1] if detection.gps_coordinates else None,
                detection.estimated_distance
            ))
        
        conn.commit()
        conn.close()
    
    def get_detections_in_area(self, center_lat: float, center_lng: float, 
                              radius_km: float = 1.0) -> List[Dict]:
        """Retrieve detections within specified area"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple bounding box query (more sophisticated spatial queries could be used)
        lat_delta = radius_km / 111.0  # Approximate degrees per km
        lng_delta = radius_km / (111.0 * math.cos(math.radians(center_lat)))
        
        cursor.execute('''
            SELECT * FROM detections 
            WHERE gps_lat BETWEEN ? AND ? 
            AND gps_lng BETWEEN ? AND ?
            AND gps_lat IS NOT NULL 
            AND gps_lng IS NOT NULL
        ''', (
            center_lat - lat_delta, center_lat + lat_delta,
            center_lng - lng_delta, center_lng + lng_delta
        ))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class MapVisualizer:
    """Create interactive maps with detected objects"""
    
    @staticmethod
    def create_detection_map(detections: List[Dict], center_lat: float, center_lng: float) -> folium.Map:
        """Create Folium map with detection markers"""
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=16,
            tiles='OpenStreetMap'
        )
        
        # Color mapping for different object types
        color_map = {
            'fence': 'brown',
            'driveway': 'gray',
            'street': 'black',
            'mailbox': 'blue',
            'fire_hydrant': 'red',
            'stop_sign': 'red',
            'traffic_light': 'yellow',
            'building': 'purple',
            'tree': 'green',
            'car': 'orange'
        }
        
        # Add markers for each detection
        for detection in detections:
            if detection['gps_lat'] and detection['gps_lng']:
                color = color_map.get(detection['class_name'], 'blue')
                
                folium.Marker(
                    location=[detection['gps_lat'], detection['gps_lng']],
                    popup=f"{detection['class_name'].title()}<br>"
                          f"Confidence: {detection['confidence']:.2f}<br>"
                          f"Distance: {detection['estimated_distance']:.1f}m",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(m)
        
        return m

class InfrastructureMapper:
    """Main application class for infrastructure mapping"""
    
    def __init__(self, google_maps_api_key: str, google_vision_credentials: str = None):
        self.maps_api = GoogleMapsAPI(google_maps_api_key)
        self.detector = ObjectDetector()
        self.data_manager = DataManager()
        
        # Load AI models
        self.detector.load_yolo_model()
        if google_vision_credentials:
            self.detector.load_google_vision_client(google_vision_credentials)
    
    def process_location(self, lat: float, lng: float, headings: List[float] = None) -> List[Detection]:
        """Process a location with multiple viewing angles"""
        if headings is None:
            headings = [0, 90, 180, 270]  # Four cardinal directions
        
        all_detections = []
        
        for heading in headings:
            logger.info(f"Processing location {lat}, {lng} with heading {heading}")
            
            # Get Street View image
            image = self.maps_api.get_street_view_image(lat, lng, heading)
            if image is None:
                continue
            
            # Create camera parameters
            camera_params = CameraParams(
                lat=lat, lng=lng, heading=heading, pitch=0
            )
            
            # Detect objects using multiple methods
            detections = []
            
            # YOLO detection
            yolo_detections = self.detector.detect_with_yolo(image)
            detections.extend(yolo_detections)
            
            # Google Vision detection (if available)
            vision_detections = self.detector.detect_with_google_vision(image)
            detections.extend(vision_detections)
            
            # Custom infrastructure detection
            custom_detections = self.detector.detect_custom_infrastructure(image)
            detections.extend(custom_detections)
            
            # Map pixel coordinates to GPS coordinates
            for detection in detections:
                gps_lat, gps_lng = CoordinateMapper.pixel_to_gps(detection, camera_params)
                detection.gps_coordinates = (gps_lat, gps_lng)
                detection.estimated_distance = CoordinateMapper.estimate_distance(detection, camera_params)
            
            # Save detections
            self.data_manager.save_detections(detections, camera_params)
            all_detections.extend(detections)
        
        return all_detections
    
    def process_address(self, address: str) -> List[Detection]:
        """Process an address for infrastructure detection"""
        lat, lng = self.maps_api.geocode_address(address)
        if lat and lng:
            return self.process_location(lat, lng)
        else:
            logger.error(f"Could not geocode address: {address}")
            return []
    
    def create_area_map(self, center_lat: float, center_lng: float, 
                       radius_km: float = 1.0) -> folium.Map:
        """Create a map showing all detections in an area"""
        detections = self.data_manager.get_detections_in_area(center_lat, center_lng, radius_km)
        return MapVisualizer.create_detection_map(detections, center_lat, center_lng)
    
    def batch_process_area(self, center_lat: float, center_lng: float, 
                          grid_spacing: float = 0.001) -> List[Detection]:
        """Process a grid of locations around a center point"""
        all_detections = []
        
        # Create grid of points
        for lat_offset in np.arange(-0.005, 0.005, grid_spacing):
            for lng_offset in np.arange(-0.005, 0.005, grid_spacing):
                lat = center_lat + lat_offset
                lng = center_lng + lng_offset
                
                detections = self.process_location(lat, lng, headings=[0, 180])
                all_detections.extend(detections)
        
        return all_detections

def main():
    """Example usage of the infrastructure mapping system"""
    
    # Configuration
    GOOGLE_MAPS_API_KEY = "your_google_maps_api_key_here"
    GOOGLE_VISION_CREDENTIALS = "path/to/google_vision_credentials.json"  # Optional
    
    # Initialize the mapper
    mapper = InfrastructureMapper(
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
        google_vision_credentials=GOOGLE_VISION_CREDENTIALS
    )
    
    # Example 1: Process a specific address
    print("Processing address...")
    detections = mapper.process_address("1600 Amphitheatre Parkway, Mountain View, CA")
    print(f"Found {len(detections)} objects")
    
    # Example 2: Process specific coordinates
    print("\nProcessing coordinates...")
    lat, lng = 37.4219999, -122.0840575  # Google headquarters
    detections = mapper.process_location(lat, lng)
    print(f"Found {len(detections)} objects at coordinates")
    
    # Example 3: Create a map with all detections
    print("\nCreating detection map...")
    detection_map = mapper.create_area_map(lat, lng, radius_km=0.5)
    detection_map.save("infrastructure_map.html")
    print("Map saved as infrastructure_map.html")
    
    # Example 4: Batch process an area
    print("\nBatch processing area...")
    all_detections = mapper.batch_process_area(lat, lng, grid_spacing=0.002)
    print(f"Total detections in area: {len(all_detections)}")
    
    # Print summary of detected objects
    object_counts = {}
    for detection in all_detections:
        obj_type = detection.class_name
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    print("\nDetected objects summary:")
    for obj_type, count in sorted(object_counts.items()):
        print(f"  {obj_type}: {count}")

if __name__ == "__main__":
    main()
