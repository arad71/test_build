import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import json
import math
from typing import List, Dict, Tuple, Optional
import logging
import base64
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# For YOLO object detection
try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("⚠️ Warning: ultralytics not installed. Install with: pip install ultralytics")

# Set page configuration
st.set_page_config(
    page_title="Street View & Satellite Obstruction Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MultiModalObstructionDetector:
    """
    A class to detect obstructions in both Google Street View and Satellite images using AI models.
    """
    
    def __init__(self, google_api_key: str):
        """
        Initialize the obstruction detector.
        
        Args:
            google_api_key: Your Google Maps API key
        """
        self.api_key = google_api_key
        self.streetview_base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.satellite_base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
        # Initialize the detection models
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')  # General purpose model
                # You could also use specialized models:
                # self.satellite_model = YOLO('yolov8n.pt')  # Could be a specialized aerial model
            except Exception as e:
                st.error(f"Error loading YOLO model: {e}")
                self.model = None
        
        # Street View obstruction categories
        self.streetview_obstruction_classes = {
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'construction': ['construction vehicle', 'crane', 'excavator'],
            'barriers': ['barrier', 'cone', 'fence'],
            'infrastructure': ['pole', 'sign', 'traffic light'],
            'people': ['person'],
            'elevation': ['building', 'wall', 'stairs', 'steps', 'hill', 'slope', 'bridge']
        }
        
        # Satellite imagery obstruction categories
        self.satellite_obstruction_classes = {
            'buildings': ['building', 'house', 'structure'],
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'aircraft', 'boat'],
            'infrastructure': ['road', 'bridge', 'parking lot', 'construction site'],
            'natural': ['tree', 'vegetation', 'water body', 'hill'],
            'commercial': ['factory', 'warehouse', 'shopping center'],
            'transportation': ['airport', 'train', 'railway', 'highway']
        }
        
        # Typical real-world sizes for objects (in meters)
        self.typical_sizes = {
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'truck': {'width': 2.5, 'height': 3.5, 'length': 8.0},
            'bus': {'width': 2.5, 'height': 3.2, 'length': 12.0},
            'building': {'width': 20.0, 'height': 15.0, 'length': 20.0},
            'house': {'width': 12.0, 'height': 8.0, 'length': 15.0},
            'tree': {'width': 8.0, 'height': 12.0, 'length': 8.0},
            'aircraft': {'width': 35.0, 'height': 12.0, 'length': 40.0},
            'boat': {'width': 4.0, 'height': 3.0, 'length': 12.0}
        }
        
        # Detection parameters
        self.min_size_threshold = 1.5
        self.min_elevation_threshold = 1.5
        self.triangle_type = "forward_cone"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_street_view_image(self, 
                            location: str, 
                            size: str = "640x640",
                            heading: Optional[int] = None,
                            pitch: int = 0,
                            fov: int = 90) -> Optional[Image.Image]:
        """Fetch a Street View image from Google's API."""
        params = {
            'location': location,
            'size': size,
            'pitch': pitch,
            'fov': fov,
            'key': self.api_key
        }
        
        if heading is not None:
            params['heading'] = heading
            
        try:
            response = requests.get(self.streetview_base_url, params=params)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            self.logger.info(f"Successfully fetched street view image for location: {location}")
            return image
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching Street View image: {e}")
            return None

    def get_satellite_image(self, 
                          location: str, 
                          zoom: int = 18,
                          size: str = "640x640",
                          maptype: str = "satellite") -> Optional[Image.Image]:
        """Fetch a satellite image from Google Maps Static API."""
        params = {
            'center': location,
            'zoom': zoom,
            'size': size,
            'maptype': maptype,
            'key': self.api_key
        }
        
        try:
            # Debug: Show the full URL being requested
            full_url = f"{self.satellite_base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            self.logger.info(f"Requesting satellite image from: {full_url}")
            
            response = requests.get(self.satellite_base_url, params=params)
            
            # Enhanced error handling
            if response.status_code == 403:
                error_msg = f"❌ API Access Forbidden (403): Maps Static API might not be enabled or API key lacks permissions"
                st.error(error_msg)
                self.logger.error(error_msg)
                return None
            elif response.status_code == 400:
                error_msg = f"❌ Bad Request (400): Check if location '{location}' is valid"
                st.error(error_msg)
                self.logger.error(error_msg)
                return None
            elif response.status_code == 402:
                error_msg = f"❌ Payment Required (402): Maps Static API requires billing to be enabled"
                st.error(error_msg)
                self.logger.error(error_msg)
                return None
            
            response.raise_for_status()
            
            # Check if response contains an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                error_msg = f"❌ Response is not an image. Content-Type: {content_type}"
                st.error(error_msg)
                self.logger.error(f"Satellite API response content: {response.text[:500]}")
                return None
            
            image = Image.open(io.BytesIO(response.content))
            self.logger.info(f"Successfully fetched satellite image for location: {location}")
            return image
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Network error fetching satellite image: {e}"
            st.error(error_msg)
            self.logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"❌ Unexpected error processing satellite image: {e}"
            st.error(error_msg)
            self.logger.error(error_msg)
            return None

    def create_detection_triangle(self, image_width: int, image_height: int) -> np.ndarray:
        """Create triangular detection area based on image dimensions."""
        if self.triangle_type == "forward_cone":
            triangle = np.array([
                [image_width // 2, image_height // 3],
                [image_width // 6, image_height - 20],
                [5 * image_width // 6, image_height - 20]
            ], np.int32)
        elif self.triangle_type == "center_up":
            triangle = np.array([
                [image_width // 2, image_height // 4],
                [image_width // 4, 3 * image_height // 4],
                [3 * image_width // 4, 3 * image_height // 4]
            ], np.int32)
        elif self.triangle_type == "full_area":
            # For satellite images, use full area detection
            triangle = np.array([
                [0, 0],
                [image_width, 0],
                [image_width, image_height],
                [0, image_height]
            ], np.int32)
        else:
            triangle = np.array([
                [image_width // 2, image_height // 3],
                [image_width // 6, image_height - 20],
                [5 * image_width // 6, image_height - 20]
            ], np.int32)
        
        return triangle

    def is_point_in_triangle(self, point: Tuple[int, int], triangle: np.ndarray) -> bool:
        """Check if a point is inside the detection area."""
        x, y = point
        result = cv2.pointPolygonTest(triangle, (float(x), float(y)), False)
        return result >= 0

    def is_detection_in_area(self, bbox: Dict, triangle: np.ndarray) -> bool:
        """Check if a detected object is within the detection area."""
        points_to_check = [
            (bbox['x1'], bbox['y1']),
            (bbox['x2'], bbox['y1']),
            (bbox['x1'], bbox['y2']),
            (bbox['x2'], bbox['y2']),
            ((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2),
        ]
        
        points_inside = sum(1 for point in points_to_check if self.is_point_in_triangle(point, triangle))
        return points_inside >= 2  # Relaxed for satellite imagery

    def estimate_object_size_satellite(self, bbox: Dict, class_name: str, zoom_level: int) -> Dict[str, float]:
        """Estimate real-world size of object from satellite imagery."""
        # Approximate meters per pixel based on zoom level
        # This is a rough approximation - actual values depend on latitude
        meters_per_pixel_zoom = {
            15: 4.77, 16: 2.39, 17: 1.19, 18: 0.60, 19: 0.30, 20: 0.15
        }
        
        meters_per_pixel = meters_per_pixel_zoom.get(zoom_level, 0.60)
        
        width_pixels = bbox['x2'] - bbox['x1']
        height_pixels = bbox['y2'] - bbox['y1']
        
        estimated_width = width_pixels * meters_per_pixel
        estimated_height = height_pixels * meters_per_pixel
        
        # Use typical length for the object type
        typical_length = self._get_typical_dimension(class_name, 'length')
        estimated_length = typical_length if typical_length else max(estimated_width, estimated_height)
        
        return {
            'width': round(estimated_width, 2),
            'height': round(estimated_height, 2),
            'length': round(estimated_length, 2)
        }

    def estimate_object_size_streetview(self, bbox: Dict, class_name: str, image_height: int) -> Dict[str, float]:
        """Estimate real-world size of object from street view."""
        # Distance estimation for street view
        object_bottom = bbox['y2']
        object_height_pixels = bbox['y2'] - bbox['y1']
        
        typical_height = self._get_typical_dimension(class_name, 'height')
        if typical_height is None:
            typical_height = 2.0
        
        horizon_line = image_height * 0.4
        ground_line = image_height * 0.9
        
        if object_bottom > ground_line:
            object_bottom = ground_line
        
        distance_factor = (object_bottom - horizon_line) / (ground_line - horizon_line)
        distance_factor = max(0.1, min(1.0, distance_factor))
        
        base_distance = (typical_height * image_height) / (object_height_pixels * 0.1)
        estimated_distance = max(1.0, min(50.0, base_distance * distance_factor))
        
        # Convert pixels to meters
        meters_per_pixel = estimated_distance * 0.001
        width_pixels = bbox['x2'] - bbox['x1']
        height_pixels = bbox['y2'] - bbox['y1']
        
        estimated_width = width_pixels * meters_per_pixel
        estimated_height = height_pixels * meters_per_pixel
        
        typical_length = self._get_typical_dimension(class_name, 'length')
        estimated_length = typical_length if typical_length else estimated_width
        
        return {
            'width': round(estimated_width, 2),
            'height': round(estimated_height, 2),
            'length': round(estimated_length, 2)
        }

    def _get_typical_dimension(self, class_name: str, dimension: str) -> Optional[float]:
        """Get typical dimension for an object class."""
        class_name_lower = class_name.lower()
        
        for obj_type, dimensions in self.typical_sizes.items():
            if obj_type in class_name_lower:
                return dimensions.get(dimension)
        
        return None

    def _is_large_enough(self, size_dict: Dict[str, float]) -> bool:
        """Check if object is larger than minimum size threshold."""
        max_dimension = max(size_dict['width'], size_dict['height'], size_dict['length'])
        return max_dimension >= self.min_size_threshold

    def _classify_obstruction_streetview(self, class_name: str) -> Optional[str]:
        """Classify if a detected object is an obstruction for street view."""
        class_name_lower = class_name.lower()
        
        for obstruction_type, classes in self.streetview_obstruction_classes.items():
            if any(cls in class_name_lower for cls in classes):
                return obstruction_type
        
        return None

    def _classify_obstruction_satellite(self, class_name: str) -> Optional[str]:
        """Classify if a detected object is an obstruction for satellite view."""
        class_name_lower = class_name.lower()
        
        for obstruction_type, classes in self.satellite_obstruction_classes.items():
            if any(cls in class_name_lower for cls in classes):
                return obstruction_type
        
        return None

    def detect_objects_yolo(self, image: np.ndarray, image_type: str = "streetview", zoom_level: int = 18) -> List[Dict]:
        """Detect objects using YOLO model."""
        if not YOLO_AVAILABLE or self.model is None:
            return []
        
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = self.model(image)
        detections = []
        image_height = image.shape[0]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Classify based on image type
                    if image_type == "streetview":
                        obstruction_type = self._classify_obstruction_streetview(class_name)
                        size_estimator = self.estimate_object_size_streetview
                    else:  # satellite
                        obstruction_type = self._classify_obstruction_satellite(class_name)
                        size_estimator = lambda bbox, class_name, _: self.estimate_object_size_satellite(bbox, class_name, zoom_level)
                    
                    if obstruction_type and confidence > 0.3:  # Lower threshold for satellite
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        
                        estimated_size = size_estimator(bbox, class_name, image_height)
                        
                        if self._is_large_enough(estimated_size):
                            detection_data = {
                                'type': obstruction_type,
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox,
                                'area': (x2 - x1) * (y2 - y1),
                                'estimated_size': estimated_size,
                                'max_dimension': max(estimated_size.values()),
                                'image_type': image_type
                            }
                            detections.append(detection_data)
        
        # Filter by detection area
        filtered_detections = self.filter_detections_by_area(detections, image.shape, image_type)
        return filtered_detections

    def filter_detections_by_area(self, detections: List[Dict], image_shape: Tuple[int, int], image_type: str) -> List[Dict]:
        """Filter detections by area (triangle for streetview, full area for satellite)."""
        if not detections:
            return detections
        
        height, width = image_shape[:2]
        
        if image_type == "satellite":
            # Use full area for satellite images
            detection_area = np.array([
                [0, 0], [width, 0], [width, height], [0, height]
            ], np.int32)
        else:
            # Use triangle for street view
            detection_area = self.create_detection_triangle(width, height)
        
        filtered_detections = []
        for detection in detections:
            if self.is_detection_in_area(detection['bbox'], detection_area):
                detection['in_detection_area'] = True
                filtered_detections.append(detection)
        
        return filtered_detections

    def visualize_detections(self, image: np.ndarray, detections: List[Dict], image_type: str = "streetview") -> np.ndarray:
        """Draw bounding boxes around detected objects."""
        if isinstance(image, Image.Image):
            annotated_image = np.array(image.copy())
        else:
            annotated_image = image.copy()
        
        # Color map for different obstruction types
        streetview_colors = {
            'vehicle': (0, 255, 0),
            'construction': (255, 0, 0),
            'barriers': (255, 255, 0),
            'infrastructure': (255, 0, 255),
            'people': (0, 255, 255),
            'elevation': (255, 165, 0)
        }
        
        satellite_colors = {
            'buildings': (255, 0, 0),
            'vehicles': (0, 255, 0),
            'infrastructure': (0, 0, 255),
            'natural': (0, 255, 0),
            'commercial': (255, 255, 0),
            'transportation': (255, 0, 255)
        }
        
        colors = streetview_colors if image_type == "streetview" else satellite_colors
        
        # Draw detection area overlay
        height, width = annotated_image.shape[:2]
        if image_type == "streetview":
            detection_area = self.create_detection_triangle(width, height)
            cv2.polylines(annotated_image, [detection_area], True, (128, 128, 128), 2)
        
        for detection in detections:
            bbox = detection['bbox']
            color = colors.get(detection['type'], (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 3)
            
            # Add label
            max_dim = detection.get('max_dimension', 0)
            label = f"{detection['class']} - {max_dim:.1f}m"
            
            cv2.putText(annotated_image, label,
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_image

    def analyze_location_multimodal(self, location: str, multiple_streetview_angles: bool = True) -> Dict:
        """Analyze a location using both street view and satellite imagery."""
        results = {
            'location': location,
            'streetview_analysis': {
                'total_obstructions': 0,
                'obstruction_types': {},
                'views': []
            },
            'satellite_analysis': {
                'total_objects': 0,
                'object_types': {},
                'detections': []
            }
        }
        
        # Street View Analysis
        if multiple_streetview_angles:
            headings = [0, 90, 180, 270]
        else:
            headings = [0]
        
        for heading in headings:
            sv_image = self.get_street_view_image(location, heading=heading)
            if sv_image is not None:
                detections = self.detect_objects_yolo(np.array(sv_image), "streetview")
                
                view_result = {
                    'heading': heading,
                    'obstructions_count': len(detections),
                    'obstructions': detections,
                    'image': sv_image
                }
                
                results['streetview_analysis']['views'].append(view_result)
                results['streetview_analysis']['total_obstructions'] += len(detections)
                
                for detection in detections:
                    obj_type = detection['type']
                    results['streetview_analysis']['obstruction_types'][obj_type] = \
                        results['streetview_analysis']['obstruction_types'].get(obj_type, 0) + 1
        
        # Satellite Analysis
        satellite_image = self.get_satellite_image(location, zoom=18)
        if satellite_image is None:
            # Try hybrid view as fallback
            self.logger.info("Satellite view failed, trying hybrid view...")
            satellite_image = self.get_satellite_image(location, zoom=18, maptype="hybrid")
            
        if satellite_image is None:
            # Try lower zoom level
            self.logger.info("High zoom failed, trying lower zoom...")
            satellite_image = self.get_satellite_image(location, zoom=16, maptype="satellite")
            
        if satellite_image is not None:
            satellite_detections = self.detect_objects_yolo(np.array(satellite_image), "satellite", zoom_level=18)
            
            results['satellite_analysis']['total_objects'] = len(satellite_detections)
            results['satellite_analysis']['detections'] = satellite_detections
            results['satellite_analysis']['image'] = satellite_image
            
            for detection in satellite_detections:
                obj_type = detection['type']
                results['satellite_analysis']['object_types'][obj_type] = \
                    results['satellite_analysis']['object_types'].get(obj_type, 0) + 1
        else:
            self.logger.warning(f"Could not fetch any satellite imagery for location: {location}")
        
        return results

def main():
    st.title("🛰️ Street View & Satellite Obstruction Detector")
    st.markdown("**Multi-Modal AI Analysis using Street View and Satellite Imagery**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Google Maps API Key",
        type="password",
        help="Enter your Google Maps API key (needs Street View Static API and Maps Static API enabled)"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Maps API key in the sidebar to continue.")
        st.info("""
        **Required APIs for this application:**
        
        1. **Street View Static API** - For street-level images
        2. **Maps Static API** - For satellite imagery
        
        **Setup Instructions:**
        1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. **Enable BOTH APIs:**
           - Street View Static API
           - Maps Static API
        4. Create credentials (API key)
        5. **Important:** Enable billing for your project (Maps Static API requires billing)
        6. Optionally restrict the API key to these specific APIs
        
        **Common Issues:**
        - ❌ Only Street View API enabled → Satellite images won't work
        - ❌ Billing not enabled → 402 Payment Required error
        - ❌ API key restrictions → 403 Forbidden error
        """)
        return
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    size_threshold = st.sidebar.slider("Size Threshold (meters)", 0.5, 10.0, 1.5, 0.1)
    
    triangle_type = st.sidebar.selectbox(
        "Street View Detection Area",
        ["forward_cone", "center_up", "center_down"],
        help="Choose the detection area shape for street view analysis"
    )
    
    multiple_angles = st.sidebar.checkbox("Multiple Street View Angles", value=True)
    satellite_zoom = st.sidebar.slider("Satellite Zoom Level", 15, 20, 18, 1)
    
    # API Testing Section
    st.sidebar.subheader("🔧 API Testing")
    if st.sidebar.button("Test API Access"):
        if api_key:
            test_location = "Times Square, New York, NY"
            
            # Test Street View API
            st.sidebar.write("Testing Street View API...")
            detector = MultiModalObstructionDetector(api_key)
            sv_test = detector.get_street_view_image(test_location)
            if sv_test:
                st.sidebar.success("✅ Street View API: Working")
            else:
                st.sidebar.error("❌ Street View API: Failed")
            
            # Test Maps Static API
            st.sidebar.write("Testing Maps Static API...")
            sat_test = detector.get_satellite_image(test_location)
            if sat_test:
                st.sidebar.success("✅ Maps Static API: Working")
            else:
                st.sidebar.error("❌ Maps Static API: Failed")
        else:
            st.sidebar.warning("Enter API key first")
    
    # Main input area
    st.header("📍 Location Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        location = st.text_input(
            "Enter an address or location:",
            placeholder="e.g., Times Square, New York, NY",
            help="Enter any address, landmark, or coordinates for multi-modal analysis"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Location", type="primary", use_container_width=True)
    
    # Analysis section
    if analyze_button and location:
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO model is not available. Please install ultralytics: `pip install ultralytics`")
            return
        
        try:
            # Initialize detector
            with st.spinner("🔧 Initializing multi-modal detector..."):
                detector = MultiModalObstructionDetector(api_key)
                detector.min_size_threshold = size_threshold
                detector.triangle_type = triangle_type
            
            # Analyze location
            with st.spinner(f"🔍 Analyzing {location} with street view and satellite imagery..."):
                results = detector.analyze_location_multimodal(location, multiple_angles)
            
            # Display results
            st.header("📊 Multi-Modal Analysis Results")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🗺️ Satellite Analysis", "🚗 Street View Analysis", "📈 Combined Statistics"])
            
            # Satellite Analysis Tab
            with tab1:
                st.subheader("🛰️ Satellite Imagery Analysis")
                
                satellite_data = results['satellite_analysis']
                
                if satellite_data.get('image') is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Satellite Image**")
                        st.image(satellite_data['image'], caption=f"Satellite View - Zoom {satellite_zoom}", use_column_width=True)
                    
                    with col2:
                        st.markdown("**Detected Objects**")
                        if satellite_data['detections']:
                            annotated_satellite = detector.visualize_detections(
                                np.array(satellite_data['image']), 
                                satellite_data['detections'],
                                "satellite"
                            )
                            st.image(annotated_satellite, caption="Satellite Object Detection", use_column_width=True)
                        else:
                            st.image(satellite_data['image'], caption="No objects detected", use_column_width=True)
                    
                    # Satellite metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Objects", satellite_data['total_objects'])
                    with col2:
                        st.metric("Object Types", len(satellite_data['object_types']))
                    with col3:
                        st.metric("Zoom Level", satellite_zoom)
                    
                    # Satellite object types
                    if satellite_data['object_types']:
                        st.subheader("🏗️ Detected Object Types (Satellite)")
                        df_satellite = pd.DataFrame(
                            list(satellite_data['object_types'].items()),
                            columns=['Type', 'Count']
                        )
                        fig_satellite = px.bar(df_satellite, x='Type', y='Count', 
                                             title="Object Distribution from Satellite View",
                                             color='Type')
                        st.plotly_chart(fig_satellite, use_container_width=True)
                    
                    # Detailed satellite detections
                    if satellite_data['detections']:
                        st.subheader("📋 Detailed Satellite Detections")
                        for i, detection in enumerate(satellite_data['detections']):
                            with st.expander(f"Object {i+1}: {detection['class'].title()}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Type:** {detection['type']}")
                                    st.write(f"**Confidence:** {detection['confidence']:.2f}")
                                    st.write(f"**Size:** {detection['max_dimension']:.1f}m")
                                with col2:
                                    st.write(f"**Width:** {detection['estimated_size']['width']}m")
                                    st.write(f"**Height:** {detection['estimated_size']['height']}m")
                                    st.write(f"**Length:** {detection['estimated_size']['length']}m")
                else:
                    st.error("❌ Could not fetch satellite imagery for this location.")
                    
                    # Troubleshooting section
                    with st.expander("🔧 Troubleshooting Satellite Image Issues", expanded=True):
                        st.markdown("""
                        **Common reasons why satellite images fail to load:**
                        
                        **1. API Configuration Issues:**
                        - ❌ Maps Static API not enabled in Google Cloud Console
                        - ❌ Only Street View Static API is enabled
                        - ✅ **Solution:** Enable Maps Static API in Google Cloud Console
                        
                        **2. Billing Requirements:**
                        - ❌ Billing not enabled for your Google Cloud project
                        - ❌ Maps Static API requires active billing (unlike some other APIs)
                        - ✅ **Solution:** Enable billing in Google Cloud Console
                        
                        **3. API Key Restrictions:**
                        - ❌ API key restricted to only Street View Static API
                        - ❌ IP address restrictions blocking requests
                        - ✅ **Solution:** Update API key restrictions to include Maps Static API
                        
                        **4. Location Issues:**
                        - ❌ Location has no high-resolution satellite imagery
                        - ❌ Location string not recognized by Google Maps
                        - ✅ **Solution:** Try a more specific address or different location
                        
                        **5. Request Limits:**
                        - ❌ Exceeded daily quota for Maps Static API
                        - ❌ Too many requests per second
                        - ✅ **Solution:** Check quota usage in Google Cloud Console
                        
                        **Quick Fix Steps:**
                        1. Use the "Test API Access" button in the sidebar
                        2. Check Google Cloud Console for both APIs enabled
                        3. Verify billing is enabled for your project
                        4. Try a well-known location like "Times Square, New York"
                        """)
                        
                        st.info("💡 **Tip:** Street View works but satellite doesn't usually means Maps Static API is not enabled or billing is not set up.")
                    
                    # Alternative: Show a map placeholder
                    st.subheader("🗺️ Location Map (Alternative)")
                    try:
                        # Create a simple map URL for reference
                        map_url = f"https://www.google.com/maps/place/{location.replace(' ', '+')}"
                        st.markdown(f"📍 [View location on Google Maps]({map_url})")
                        
                        # You could also use Folium or other mapping libraries here
                        st.info("Consider using Folium or other mapping libraries as an alternative for basic location visualization.")
                    except:
                        pass
            
            # Street View Analysis Tab
            with tab2:
                st.subheader("🚗 Street View Analysis")
                
                streetview_data = results['streetview_analysis']
                
                if not streetview_data['views']:
                    st.error("❌ Could not fetch Street View images for this location.")
                else:
                    # Street view metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Obstructions", streetview_data['total_obstructions'])
                    with col2:
                        st.metric("Views Analyzed", len(streetview_data['views']))
                    with col3:
                        st.metric("Obstruction Types", len(streetview_data['obstruction_types']))
                    
                    # Street view analysis by angle
                    for i, view in enumerate(streetview_data['views']):
                        with st.expander(f"Street View {view['heading']}° - {view['obstructions_count']} obstructions", expanded=(i==0)):
                            
                            if view['image'] is not None:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Original Street View**")
                                    st.image(view['image'], caption=f"Heading {view['heading']}°", use_column_width=True)
                                
                                with col2:
                                    st.markdown("**With Detections**")
                                    if view['obstructions']:
                                        annotated_image = detector.visualize_detections(
                                            np.array(view['image']), 
                                            view['obstructions'],
                                            "streetview"
                                        )
                                        st.image(annotated_image, caption="Detected Obstructions", use_column_width=True)
                                    else:
                                        st.image(view['image'], caption="No obstructions detected", use_column_width=True)
                            
                            # Obstruction details
                            if view['obstructions']:
                                st.markdown("**Detected Obstructions:**")
                                for j, obs in enumerate(view['obstructions']):
                                    st.write(f"**{j+1}. {obs['class'].title()}** ({obs['type']})")
                                    st.write(f"   • Confidence: {obs['confidence']:.2f}")
                                    st.write(f"   • Size: {obs['max_dimension']:.1f}m")
                                    st.write(f"   • Dimensions: {obs['estimated_size']['width']}×{obs['estimated_size']['height']}×{obs['estimated_size']['length']}m")
            
            # Combined Statistics Tab
            with tab3:
                st.subheader("📈 Combined Analysis Statistics")
                
                # Overall metrics
                total_street_objects = streetview_data['total_obstructions']
                total_satellite_objects = satellite_data['total_objects']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Street View Objects", total_street_objects)
                with col2:
                    st.metric("Satellite Objects", total_satellite_objects)
                with col3:
                    st.metric("Total Combined", total_street_objects + total_satellite_objects)
                with col4:
                    st.metric("Analysis Coverage", "360° + Aerial")
                
                # Combined visualization
                if streetview_data['obstruction_types'] or satellite_data['object_types']:
                    # Create combined data for comparison
                    combined_data = []
                    
                    for obj_type, count in streetview_data['obstruction_types'].items():
                        combined_data.append({'Source': 'Street View', 'Type': obj_type, 'Count': count})
                    
                    for obj_type, count in satellite_data['object_types'].items():
                        combined_data.append({'Source': 'Satellite', 'Type': obj_type, 'Count': count})
                    
                    if combined_data:
                        df_combined = pd.DataFrame(combined_data)
                        fig_combined = px.bar(df_combined, x='Type', y='Count', color='Source',
                                            title="Object Detection Comparison: Street View vs Satellite",
                                            barmode='group')
                        st.plotly_chart(fig_combined, use_container_width=True)
                
                # Download section
                st.subheader("💾 Download Results")
                
                # Generate comprehensive report
                report_text = f"""
Multi-Modal Obstruction Analysis Report
=====================================
Location: {results['location']}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

STREET VIEW ANALYSIS:
- Total Obstructions: {streetview_data['total_obstructions']}
- Views Analyzed: {len(streetview_data['views'])}
- Obstruction Types: {dict(streetview_data['obstruction_types'])}

SATELLITE ANALYSIS:
- Total Objects: {satellite_data['total_objects']}
- Zoom Level: {satellite_zoom}
- Object Types: {dict(satellite_data['object_types'])}

DETECTION PARAMETERS:
- Size Threshold: {size_threshold}m
- Street View Detection Area: {triangle_type}
- Multiple Angles: {multiple_angles}

COMBINED INSIGHTS:
- Total Objects Detected: {total_street_objects + total_satellite_objects}
- Analysis Coverage: 360° Street View + Aerial Satellite View
                """
                
                st.download_button(
                    label="📄 Download Comprehensive Report",
                    data=report_text,
                    file_name=f"multimodal_analysis_{location.replace(' ', '_').replace(',', '')}.txt",
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
