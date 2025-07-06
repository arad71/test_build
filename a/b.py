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
    page_title="Street View Obstruction Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreetViewObstructionDetector:
    """
    A class to detect obstructions in Google Street View images using AI models.
    """
    
    def __init__(self, google_api_key: str, model_type: str = "yolo"):
        """
        Initialize the obstruction detector.
        
        Args:
            google_api_key: Your Google Street View Static API key
            model_type: Type of model to use ("yolo" or "google_vision")
        """
        self.api_key = google_api_key
        self.model_type = model_type
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        
        # Initialize the detection model
        if model_type == "yolo" and YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n.pt')  # Downloads pre-trained model
            except Exception as e:
                st.error(f"Error loading YOLO model: {e}")
                self.model = None
        
        # Define obstruction categories
        self.obstruction_classes = {
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'construction': ['construction vehicle', 'crane', 'excavator'],
            'barriers': ['barrier', 'cone', 'fence'],
            'infrastructure': ['pole', 'sign', 'traffic light'],
            'people': ['person'],
            'elevation': ['building', 'wall', 'stairs', 'steps', 'hill', 'slope', 'bridge']
        }
        
        # Typical real-world sizes for objects (in meters) for size estimation
        self.typical_sizes = {
            'car': {'width': 1.8, 'height': 1.5, 'length': 4.5},
            'truck': {'width': 2.5, 'height': 3.5, 'length': 8.0},
            'bus': {'width': 2.5, 'height': 3.2, 'length': 12.0},
            'motorcycle': {'width': 0.8, 'height': 1.2, 'length': 2.2},
            'bicycle': {'width': 0.6, 'height': 1.1, 'length': 1.8},
            'person': {'width': 0.5, 'height': 1.7, 'length': 0.3},
            'pole': {'width': 0.2, 'height': 6.0, 'length': 0.2},
            'sign': {'width': 1.0, 'height': 1.0, 'length': 0.1},
            'building': {'width': 10.0, 'height': 15.0, 'length': 10.0},
            'wall': {'width': 5.0, 'height': 3.0, 'length': 0.3},
            'stairs': {'width': 2.0, 'height': 2.0, 'length': 3.0},
            'bridge': {'width': 10.0, 'height': 8.0, 'length': 20.0}
        }
        
        # Street View camera parameters
        self.camera_height = 2.5  # meters above ground
        self.image_width = 640
        self.image_height = 640
        self.min_size_threshold = 1.5  # minimum size in meters to detect
        self.min_elevation_threshold = 1.5  # minimum elevation difference in meters
        
        # Triangular detection area configuration
        self.detection_triangle = None
        self.triangle_type = "forward_cone"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_street_view_image(self, 
                            location: str, 
                            size: str = "640x640",
                            heading: Optional[int] = None,
                            pitch: int = 0,
                            fov: int = 90) -> Optional[np.ndarray]:
        """
        Fetch a Street View image from Google's API.
        """
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
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            self.logger.info(f"Successfully fetched image for location: {location}")
            return image
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching Street View image: {e}")
            return None

    def create_detection_triangle(self, image_width: int, image_height: int) -> np.ndarray:
        """Create triangular detection area based on image dimensions."""
        if self.triangle_type == "forward_cone":
            triangle = np.array([
                [image_width // 2, image_height // 3],           # Top center
                [image_width // 6, image_height - 20],           # Bottom left
                [5 * image_width // 6, image_height - 20]        # Bottom right
            ], np.int32)
        elif self.triangle_type == "center_up":
            triangle = np.array([
                [image_width // 2, image_height // 4],
                [image_width // 4, 3 * image_height // 4],
                [3 * image_width // 4, 3 * image_height // 4]
            ], np.int32)
        else:  # Default to forward cone
            triangle = np.array([
                [image_width // 2, image_height // 3],
                [image_width // 6, image_height - 20],
                [5 * image_width // 6, image_height - 20]
            ], np.int32)
        
        return triangle

    def is_point_in_triangle(self, point: Tuple[int, int], triangle: np.ndarray) -> bool:
        """Check if a point is inside the triangular detection area."""
        x, y = point
        result = cv2.pointPolygonTest(triangle, (float(x), float(y)), False)
        return result >= 0

    def is_detection_in_triangle(self, bbox: Dict, triangle: np.ndarray) -> bool:
        """Check if a detected object is within the triangular detection area."""
        points_to_check = [
            (bbox['x1'], bbox['y1']),
            (bbox['x2'], bbox['y1']),
            (bbox['x1'], bbox['y2']),
            (bbox['x2'], bbox['y2']),
            ((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2),
            ((bbox['x1'] + bbox['x2']) // 2, bbox['y1']),
            ((bbox['x1'] + bbox['x2']) // 2, bbox['y2']),
        ]
        
        points_inside = sum(1 for point in points_to_check if self.is_point_in_triangle(point, triangle))
        return points_inside >= 3

    def estimate_object_distance(self, bbox: Dict, class_name: str, image_height: int) -> float:
        """Estimate distance to object using object size and position in image."""
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
        estimated_distance = base_distance * distance_factor
        
        return max(1.0, min(50.0, estimated_distance))

    def estimate_object_size(self, bbox: Dict, class_name: str, distance: float) -> Dict[str, float]:
        """Estimate real-world size of object."""
        width_pixels = bbox['x2'] - bbox['x1']
        height_pixels = bbox['y2'] - bbox['y1']
        
        meters_per_pixel = distance * 0.001
        
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

    def _classify_obstruction(self, class_name: str) -> Optional[str]:
        """Classify if a detected object is an obstruction."""
        class_name_lower = class_name.lower()
        
        for obstruction_type, classes in self.obstruction_classes.items():
            if any(cls in class_name_lower for cls in classes):
                return obstruction_type
        
        return None

    def detect_obstructions_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect obstructions using YOLO model within triangular detection area."""
        if not YOLO_AVAILABLE or self.model is None:
            return []
        
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = self.model(image)
        obstructions = []
        image_height = image.shape[0]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    obstruction_type = self._classify_obstruction(class_name)
                    if obstruction_type and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        
                        distance = self.estimate_object_distance(bbox, class_name, image_height)
                        estimated_size = self.estimate_object_size(bbox, class_name, distance)
                        
                        if self._is_large_enough(estimated_size):
                            obstruction_data = {
                                'type': obstruction_type,
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox,
                                'area': (x2 - x1) * (y2 - y1),
                                'distance': round(distance, 2),
                                'estimated_size': estimated_size,
                                'max_dimension': max(estimated_size.values())
                            }
                            obstructions.append(obstruction_data)
        
        # Filter by triangular detection area
        filtered_obstructions = self.filter_detections_by_triangle(obstructions, image.shape)
        return filtered_obstructions

    def filter_detections_by_triangle(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Filter detections to only include those within the triangular detection area."""
        if not detections:
            return detections
        
        height, width = image_shape[:2]
        triangle = self.create_detection_triangle(width, height)
        
        filtered_detections = []
        for detection in detections:
            if self.is_detection_in_triangle(detection['bbox'], triangle):
                detection['in_triangle_area'] = True
                filtered_detections.append(detection)
        
        return filtered_detections

    def visualize_obstructions(self, image: np.ndarray, obstructions: List[Dict]) -> np.ndarray:
        """Draw bounding boxes around detected obstructions."""
        if isinstance(image, Image.Image):
            annotated_image = np.array(image.copy())
        else:
            annotated_image = image.copy()
        
        # Color map for different obstruction types
        colors = {
            'vehicle': (0, 255, 0),      # Green
            'construction': (255, 0, 0),  # Red
            'barriers': (255, 255, 0),    # Yellow
            'infrastructure': (255, 0, 255),  # Magenta
            'people': (0, 255, 255),      # Cyan
            'elevation': (255, 165, 0)    # Orange
        }
        
        # Draw triangle area
        height, width = annotated_image.shape[:2]
        triangle = self.create_detection_triangle(width, height)
        cv2.polylines(annotated_image, [triangle], True, (128, 128, 128), 2)
        cv2.fillPoly(annotated_image, [triangle], (128, 128, 128, 50))
        
        for obs in obstructions:
            bbox = obs['bbox']
            color = colors.get(obs['type'], (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 3)
            
            # Add label
            size_info = obs.get('estimated_size', {})
            max_dim = obs.get('max_dimension', 0)
            label = f"{obs['class']} - {max_dim:.1f}m"
                
            cv2.putText(annotated_image, label,
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_image

    def analyze_location(self, location: str, multiple_views: bool = True) -> Dict:
        """Analyze a location for obstructions from multiple viewpoints."""
        results = {
            'location': location,
            'total_obstructions': 0,
            'obstruction_types': {},
            'views': []
        }
        
        headings = [0, 90, 180, 270] if multiple_views else [0]
        
        for i, heading in enumerate(headings):
            image = self.get_street_view_image(location, heading=heading)
            if image is None:
                continue
            
            obstructions = self.detect_obstructions_yolo(np.array(image))
            
            view_result = {
                'heading': heading,
                'obstructions_count': len(obstructions),
                'obstructions': obstructions,
                'image': image
            }
            
            results['views'].append(view_result)
            results['total_obstructions'] += len(obstructions)
            
            for obs in obstructions:
                obs_type = obs['type']
                results['obstruction_types'][obs_type] = results['obstruction_types'].get(obs_type, 0) + 1
            
        return results

def main():
    st.title("🔍 Street View Obstruction Detector")
    st.markdown("**AI-Powered Analysis of Urban Navigation Obstacles using Computer Vision**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Google Street View API Key",
        type="password",
        help="Enter your Google Street View Static API key"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Google Street View API key in the sidebar to continue.")
        st.info("""
        To get a Google Street View API key:
        1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select an existing one
        3. Enable the Street View Static API
        4. Create credentials (API key)
        5. Optionally restrict the API key to Street View Static API
        """)
        return
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    size_threshold = st.sidebar.slider("Size Threshold (meters)", 0.5, 5.0, 1.5, 0.1)
    elevation_threshold = st.sidebar.slider("Elevation Threshold (meters)", 0.5, 5.0, 1.5, 0.1)
    
    triangle_type = st.sidebar.selectbox(
        "Detection Area",
        ["forward_cone", "center_up", "center_down"],
        help="Choose the triangular detection area shape"
    )
    
    multiple_views = st.sidebar.checkbox("Multiple Viewing Angles", value=True)
    
    # Main input area
    st.header("📍 Location Input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        location = st.text_input(
            "Enter an address or location:",
            placeholder="e.g., Times Square, New York, NY or 101 Soldiers Rd, Roleystone WA 6111, Australia",
            help="Enter any address, landmark, or coordinates"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze Location", type="primary", use_container_width=True)
    
    # Example locations
    st.subheader("📋 Example Locations")
    example_cols = st.columns(4)
    
    examples = [
        "Times Square, New York, NY",
        "Golden Gate Bridge, San Francisco, CA",
        "Parliament House, Canberra, Australia",
        "Big Ben, London, UK"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"📍 {example.split(',')[0]}", use_container_width=True):
                location = example
                analyze_button = True
    
    # Analysis section
    if analyze_button and location:
        if not YOLO_AVAILABLE:
            st.error("❌ YOLO model is not available. Please install ultralytics: `pip install ultralytics`")
            return
        
        try:
            # Initialize detector
            with st.spinner("🔧 Initializing detector..."):
                detector = StreetViewObstructionDetector(api_key)
                detector.min_size_threshold = size_threshold
                detector.min_elevation_threshold = elevation_threshold
                detector.triangle_type = triangle_type
            
            # Analyze location
            with st.spinner(f"🔍 Analyzing {location}..."):
                results = detector.analyze_location(location, multiple_views)
            
            if not results['views']:
                st.error("❌ Could not fetch Street View images for this location. Please check the address and try again.")
                return
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Obstructions", results['total_obstructions'])
            with col2:
                st.metric("Views Analyzed", len(results['views']))
            with col3:
                st.metric("Size Threshold", f"{size_threshold}m")
            with col4:
                st.metric("Detection Types", len(results['obstruction_types']))
            
            # Obstruction types chart
            if results['obstruction_types']:
                st.subheader("🏷️ Obstruction Types Distribution")
                
                df_types = pd.DataFrame(
                    list(results['obstruction_types'].items()),
                    columns=['Type', 'Count']
                )
                
                fig = px.pie(df_types, values='Count', names='Type', title="Distribution of Obstruction Types")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            # View-by-view analysis
            st.subheader("🔄 Multi-Angle Analysis")
            
            for i, view in enumerate(results['views']):
                with st.expander(f"View {i+1} - Heading {view['heading']}° ({view['obstructions_count']} obstructions)", expanded=(i==0)):
                    
                    if view['image'] is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Original Image**")
                            st.image(view['image'], caption=f"Street View - Heading {view['heading']}°", use_column_width=True)
                        
                        with col2:
                            st.markdown("**With Detections**")
                            if view['obstructions']:
                                annotated_image = detector.visualize_obstructions(
                                    np.array(view['image']), 
                                    view['obstructions']
                                )
                                st.image(annotated_image, caption="Detected Obstructions", use_column_width=True)
                            else:
                                st.image(view['image'], caption="No obstructions detected", use_column_width=True)
                    
                    # Detailed obstruction list
                    if view['obstructions']:
                        st.markdown("**Detected Obstructions:**")
                        
                        for j, obs in enumerate(view['obstructions']):
                            with st.container():
                                st.markdown(f"""
                                **{j+1}. {obs['class'].title()}** ({obs['type']})
                                - Confidence: {obs['confidence']:.2f}
                                - Size: {obs['max_dimension']:.1f}m
                                - Distance: ~{obs['distance']:.1f}m
                                - Dimensions: W:{obs['estimated_size']['width']}m × H:{obs['estimated_size']['height']}m × L:{obs['estimated_size']['length']}m
                                """)
                    else:
                        st.info("No obstructions detected in this view within the detection area.")
            
            # Download section
            st.subheader("💾 Download Results")
            
            # Generate report
            report_text = f"""
Street View Obstruction Analysis Report
=====================================
Location: {results['location']}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Obstructions: {results['total_obstructions']}
Views Analyzed: {len(results['views'])}
Size Threshold: {size_threshold}m
Elevation Threshold: {elevation_threshold}m
Detection Area: {triangle_type}

Obstruction Types:
{chr(10).join([f"- {k}: {v}" for k, v in results['obstruction_types'].items()]) if results['obstruction_types'] else "None detected"}

Detailed Analysis:
{chr(10).join([f"View {i+1} (Heading {view['heading']}°): {view['obstructions_count']} obstructions" for i, view in enumerate(results['views'])])}
            """
            
            st.download_button(
                label="📄 Download Analysis Report",
                data=report_text,
                file_name=f"obstruction_analysis_{location.replace(' ', '_').replace(',', '')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
