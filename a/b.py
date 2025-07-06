import requests
import cv2
import numpy as np
from PIL import Image
import io
import json
import math
from typing import List, Dict, Tuple, Optional
import logging

# For YOLO object detection
try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

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
        self.api_key = 'AIzaSyB53HJkSXyesczASdc6kPv7Gv5av20rUwA' #google_api_key
        self.model_type = 'yolo' #model_type
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        
        # Initialize the detection model
        if model_type == "yolo" and YOLO_AVAILABLE:
            self.model = YOLO('yolov8n.pt')  # Downloads pre-trained model
        elif model_type == "google_vision":
            try:
                from google.cloud import vision
                self.vision_client = vision.ImageAnnotatorClient()
            except ImportError:
                raise ImportError("Install google-cloud-vision: pip install google-cloud-vision")
        
        # Define obstruction categories (you can customize these)
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
        
        # Street View camera parameters (approximate)
        self.camera_height = 2.5  # meters above ground
        self.image_width = 640
        self.image_height = 640
        self.min_size_threshold = 1.5  # minimum size in meters to detect
        self.min_elevation_threshold = 1.5  # minimum elevation difference in meters
        
        # Triangular detection area configuration
        self.detection_triangle = None  # Will be set based on image dimensions
        self.triangle_type = "forward_cone"  # Options: "forward_cone", "custom", "center_up", "center_down"
        
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
        
        Args:
            location: Address or lat,lng coordinates
            size: Image size (e.g., "640x640")
            heading: Compass heading (0-360)
            pitch: Up/down angle (-90 to 90)
            fov: Field of view (10-120)
            
        Returns:
            Image as numpy array or None if failed
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
            
        # try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)
            
            self.logger.info(f"Successfully fetched image for location: {location}")
            return image

    def visualize_with_triangle_types(self, 
                                    image: np.ndarray, 
                                    obstructions: List[Dict],
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw different triangle types based on obstruction categories.
        
        Args:
            image: Original image
            obstructions: List of detected obstructions
            save_path: Optional path to save the annotated image
            
        Returns:
            Annotated image with type-specific triangular markers
        """
        annotated_image = image.copy()
        
        # Color and triangle variant mapping
        triangle_config = {
            'vehicle': {'color': (0, 255, 0), 'variant': 'downward'},      # Green downward
            'construction': {'color': (255, 0, 0), 'variant': 'upward'},   # Red upward
            'barriers': {'color': (255, 255, 0), 'variant': 'left'},       # Yellow left
            'infrastructure': {'color': (255, 0, 255), 'variant': 'right'}, # Magenta right
            'people': {'color': (0, 255, 255), 'variant': 'downward'},     # Cyan downward
            'elevation': {'color': (128, 0, 255), 'variant': 'upward'}     # Purple upward
        }
        
        for obs in obstructions:
            bbox = obs['bbox']
            obs_type = obs['type']
            config = triangle_config.get(obs_type, {'color': (128, 128, 128), 'variant': 'upward'})
            
            # Draw type-specific triangle
            annotated_image = self.draw_triangle_variant(
                annotated_image, bbox, config['color'], config['variant']
            )
            
            # Add labels and details (same as before)
            size_info = obs.get('estimated_size', {})
            max_dim = obs.get('max_dimension', 0)
            elevation = obs.get('elevation_difference', 0)
            
            if size_info and elevation > 0:
                label = f"{obs['class']} - {max_dim:.1f}m, +{elevation:.1f}m"
            elif size_info:
                label = f"{obs['class']} - {max_dim:.1f}m"
            elif elevation > 0:
                label = f"{obs['class']} - +{elevation:.1f}m elev"
            else:
                label = f"{obs['class']}"
            
            # Position label based on triangle orientation
            if config['variant'] == 'upward':
                label_y = bbox['y1'] - 15
            elif config['variant'] == 'downward':
                label_y = bbox['y2'] + 25
            else:  # left or right
                label_y = bbox['y1'] - 10
            
            label_x = bbox['x1']
            
            # Add background for better readability
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, 
                         (label_x - 2, label_y - label_size[1] - 4),
                         (label_x + label_size[0] + 2, label_y + 4),
                         (0, 0, 0), -1)
            
            cv2.putText(annotated_image, label,
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 2)
        
        # Add legend
        annotated_image = self._add_triangle_legend(annotated_image, triangle_config)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Annotated image with triangle types saved to: {save_path}")
        
        return annotated_image

    def _add_triangle_legend(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """
        Add a legend showing triangle types and their meanings.
        
        Args:
            image: Image to add legend to
            config: Triangle configuration mapping
            
        Returns:
            Image with legend added
        """
        height, width = image.shape[:2]
        legend_x = width - 250
        legend_y = 30
        
        # Legend background
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (width - 10, legend_y + len(config) * 30 + 20), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (width - 10, legend_y + len(config) * 30 + 20), 
                     (255, 255, 255), 2)
        
        # Legend title
        cv2.putText(image, "Triangle Legend:", 
                   (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend entries
        for i, (obs_type, type_config) in enumerate(config.items()):
            y_pos = legend_y + 25 + i * 25
            
            # Draw small triangle example
            small_bbox = {
                'x1': legend_x, 'y1': y_pos - 8,
                'x2': legend_x + 16, 'y2': y_pos + 8
            }
            self.draw_triangle_variant(image, small_bbox, type_config['color'], type_config['variant'])
            
            # Type name and variant
            variant_symbols = {
                'upward': '▲', 'downward': '▼', 
                'left': '◄', 'right': '►'
            }
            symbol = variant_symbols.get(type_config['variant'], '▲')
            
            cv2.putText(image, f"{symbol} {obs_type.title()}", 
                       (legend_x + 25, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image_array
            
        # except requests.exceptions.RequestException as e:
        #     self.logger.error(f"Error fetching Street View image: {e}")
        #     return None

    def set_detection_triangle(self, triangle_type: str = "forward_cone", custom_points: Optional[List[Tuple[int, int]]] = None):
        """
        Set the triangular detection area.
        
        Args:
            triangle_type: Type of triangle ("forward_cone", "center_up", "center_down", "left_sector", "right_sector", "custom")
            custom_points: For custom triangle, provide 3 points as [(x1,y1), (x2,y2), (x3,y3)]
        """
        self.triangle_type = triangle_type
        
        # Will be updated when processing each image based on actual dimensions
        self.detection_triangle = None
        
        if triangle_type == "custom" and custom_points:
            if len(custom_points) == 3:
                self.detection_triangle = np.array(custom_points, np.int32)
            else:
                self.logger.warning("Custom triangle requires exactly 3 points")
                self.triangle_type = "forward_cone"
        
        self.logger.info(f"Detection triangle set to: {triangle_type}")

    def create_detection_triangle(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Create triangular detection area based on image dimensions.
        
        Args:
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            Triangle points as numpy array
        """
        if self.triangle_type == "forward_cone":
            # Forward-facing cone (like vehicle's forward vision)
            # Wide at bottom, narrow at top
            triangle = np.array([
                [image_width // 2, image_height // 3],           # Top center (horizon area)
                [image_width // 6, image_height - 20],           # Bottom left
                [5 * image_width // 6, image_height - 20]        # Bottom right
            ], np.int32)
            
        elif self.triangle_type == "center_up":
            # Triangle pointing upward from center
            triangle = np.array([
                [image_width // 2, image_height // 4],           # Top center
                [image_width // 4, 3 * image_height // 4],       # Bottom left
                [3 * image_width // 4, 3 * image_height // 4]    # Bottom right
            ], np.int32)
            
        elif self.triangle_type == "center_down":
            # Triangle pointing downward from center
            triangle = np.array([
                [image_width // 4, image_height // 4],           # Top left
                [3 * image_width // 4, image_height // 4],       # Top right
                [image_width // 2, 3 * image_height // 4]        # Bottom center
            ], np.int32)
            
        elif self.triangle_type == "left_sector":
            # Left side triangular sector
            triangle = np.array([
                [0, 0],                                          # Top left corner
                [0, image_height],                               # Bottom left corner
                [image_width // 2, image_height // 2]           # Center right
            ], np.int32)
            
        elif self.triangle_type == "right_sector":
            # Right side triangular sector
            triangle = np.array([
                [image_width, 0],                                # Top right corner
                [image_width, image_height],                     # Bottom right corner
                [image_width // 2, image_height // 2]           # Center left
            ], np.int32)
            
        elif self.triangle_type == "custom" and self.detection_triangle is not None:
            # Use custom triangle (already set)
            triangle = self.detection_triangle
            
        else:
            # Default to forward cone
            triangle = np.array([
                [image_width // 2, image_height // 3],
                [image_width // 6, image_height - 20],
                [5 * image_width // 6, image_height - 20]
            ], np.int32)
        
        return triangle

    def is_point_in_triangle(self, point: Tuple[int, int], triangle: np.ndarray) -> bool:
        """
        Check if a point is inside the triangular detection area.
        
        Args:
            point: Point coordinates (x, y)
            triangle: Triangle vertices as numpy array
            
        Returns:
            True if point is inside triangle
        """
        x, y = point
        
        # Use OpenCV's pointPolygonTest function
        result = cv2.pointPolygonTest(triangle, (float(x), float(y)), False)
        return result >= 0  # >= 0 means inside or on boundary

    def is_detection_in_triangle(self, bbox: Dict, triangle: np.ndarray) -> bool:
        """
        Check if a detected object (bounding box) is within the triangular detection area.
        
        Args:
            bbox: Bounding box with x1, y1, x2, y2 coordinates
            triangle: Triangle vertices as numpy array
            
        Returns:
            True if object is significantly within the triangle
        """
        # Check multiple points of the bounding box
        points_to_check = [
            (bbox['x1'], bbox['y1']),                           # Top-left
            (bbox['x2'], bbox['y1']),                           # Top-right
            (bbox['x1'], bbox['y2']),                           # Bottom-left
            (bbox['x2'], bbox['y2']),                           # Bottom-right
            ((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2),  # Center
            ((bbox['x1'] + bbox['x2']) // 2, bbox['y1']),       # Top-center
            ((bbox['x1'] + bbox['x2']) // 2, bbox['y2']),       # Bottom-center
        ]
        
        # Count how many points are inside the triangle
        points_inside = sum(1 for point in points_to_check if self.is_point_in_triangle(point, triangle))
        
        # Consider detection valid if at least 3 out of 7 points are inside
        # This allows for partial overlaps while filtering out mostly outside objects
        return points_inside >= 3

    def filter_detections_by_triangle(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Filter detections to only include those within the triangular detection area.
        
        Args:
            detections: List of detected objects
            image_shape: Image dimensions (height, width)
            
        Returns:
            Filtered list of detections within triangle
        """
        if not detections:
            return detections
        
        height, width = image_shape[:2]
        triangle = self.create_detection_triangle(width, height)
        
        filtered_detections = []
        for detection in detections:
            if self.is_detection_in_triangle(detection['bbox'], triangle):
                # Add triangle area info to detection
                detection['in_triangle_area'] = True
                filtered_detections.append(detection)
            else:
                # Log filtered out detection
                self.logger.debug(f"Filtered out {detection['class']} - outside triangle area")
        
        self.logger.info(f"Triangle filtering: {len(filtered_detections)}/{len(detections)} detections kept")
        return filtered_detections

    def estimate_object_distance(self, bbox: Dict, class_name: str, image_height: int) -> float:
        """
        Estimate distance to object using object size and position in image.
        
        Args:
            bbox: Bounding box coordinates
            class_name: Detected object class name
            image_height: Height of the image in pixels
            
        Returns:
            Estimated distance in meters
        """
        # Get object bottom position (closer to ground = further away due to perspective)
        object_bottom = bbox['y2']
        object_height_pixels = bbox['y2'] - bbox['y1']
        
        # Get typical real-world height for this object type
        typical_height = self._get_typical_dimension(class_name, 'height')
        if typical_height is None:
            typical_height = 2.0  # Default assumption
        
        # Simple perspective-based distance estimation
        # Objects lower in the image are generally further away
        horizon_line = image_height * 0.4  # Approximate horizon line
        ground_line = image_height * 0.9   # Approximate ground line
        
        # Calculate relative position between horizon and ground
        if object_bottom > ground_line:
            object_bottom = ground_line
        
        distance_factor = (object_bottom - horizon_line) / (ground_line - horizon_line)
        distance_factor = max(0.1, min(1.0, distance_factor))
        
        # Estimate distance using object height in pixels vs real height
        # Closer objects appear larger
        base_distance = (typical_height * image_height) / (object_height_pixels * 0.1)
        estimated_distance = base_distance * distance_factor
        
        return max(1.0, min(50.0, estimated_distance))  # Clamp between 1-50 meters

    def estimate_object_size(self, bbox: Dict, class_name: str, distance: float) -> Dict[str, float]:
        """
        Estimate real-world size of object.
        
        Args:
            bbox: Bounding box coordinates
            class_name: Detected object class name
            distance: Estimated distance to object
            
        Returns:
            Dictionary with estimated width, height, length in meters
        """
        # Calculate pixel dimensions
        width_pixels = bbox['x2'] - bbox['x1']
        height_pixels = bbox['y2'] - bbox['y1']
        
        # Rough conversion from pixels to meters based on distance
        # This is approximate and assumes specific camera parameters
        meters_per_pixel = distance * 0.001  # Rough approximation
        
        estimated_width = width_pixels * meters_per_pixel
        estimated_height = height_pixels * meters_per_pixel
        
        # Use typical length for the object type
        typical_length = self._get_typical_dimension(class_name, 'length')
        estimated_length = typical_length if typical_length else estimated_width
        
        return {
            'width': round(estimated_width, 2),
            'height': round(estimated_height, 2), 
            'length': round(estimated_length, 2)
        }

    def _get_typical_dimension(self, class_name: str, dimension: str) -> Optional[float]:
        """
        Get typical dimension for an object class.
        
        Args:
            class_name: Name of the object class
            dimension: 'width', 'height', or 'length'
            
        Returns:
            Typical dimension in meters or None
        """
        class_name_lower = class_name.lower()
        
        for obj_type, dimensions in self.typical_sizes.items():
            if obj_type in class_name_lower:
                return dimensions.get(dimension)
        
        return None

    def _is_large_enough(self, size_dict: Dict[str, float]) -> bool:
        """
        Check if object is larger than minimum size threshold.
        
        Args:
            size_dict: Dictionary with width, height, length
            
        Returns:
            True if any dimension exceeds threshold
        """
        max_dimension = max(size_dict['width'], size_dict['height'], size_dict['length'])
        return max_dimension >= self.min_size_threshold

    def detect_ground_plane(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect the ground plane and horizon line in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with ground plane information
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect horizontal edges (likely ground/building boundaries)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find horizontal lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=width*0.3, maxLineGap=20)
        
        # Estimate horizon and ground lines
        horizon_line = height * 0.4  # Default horizon estimate
        ground_line = height * 0.85   # Default ground estimate
        
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    horizontal_lines.append((y1 + y2) / 2)
            
            if horizontal_lines:
                horizontal_lines.sort()
                # Estimate horizon (upper third) and ground (lower portion)
                upper_lines = [y for y in horizontal_lines if y < height * 0.6]
                lower_lines = [y for y in horizontal_lines if y > height * 0.6]
                
                if upper_lines:
                    horizon_line = np.median(upper_lines)
                if lower_lines:
                    ground_line = np.median(lower_lines)
        
        return {
            'horizon_y': horizon_line,
            'ground_y': ground_line,
            'image_height': height,
            'ground_plane_angle': 0  # Simplified assumption
        }

    def estimate_elevation_difference(self, bbox: Dict, ground_info: Dict, class_name: str) -> float:
        """
        Estimate elevation difference of an object from ground level.
        
        Args:
            bbox: Bounding box coordinates
            ground_info: Ground plane information
            class_name: Object class name
            
        Returns:
            Estimated elevation difference in meters
        """
        object_bottom = bbox['y2']
        object_top = bbox['y1']
        object_height = object_bottom - object_top
        
        ground_level = ground_info['ground_y']
        horizon_level = ground_info['horizon_y']
        
        # Calculate how much object extends above expected ground level
        ground_clearance = ground_level - object_bottom
        object_above_ground = ground_level - object_top
        
        # Convert pixel differences to approximate meters
        # Objects higher in the image are generally further away or elevated
        vertical_scale = self.camera_height / (ground_level - horizon_level) if ground_level != horizon_level else 0.01
        
        # Estimate elevation based on object position relative to ground
        if object_bottom < ground_level:  # Object appears above ground line
            elevation_pixels = ground_level - object_bottom
            estimated_elevation = elevation_pixels * vertical_scale
            
            # For buildings and walls, add height estimation
            if any(keyword in class_name.lower() for keyword in ['building', 'wall', 'bridge']):
                object_height_meters = object_height * vertical_scale
                estimated_elevation += object_height_meters * 0.5  # Adjust for perspective
            
            return max(0, estimated_elevation)
        else:
            # Object at or below ground level
            return 0

    def detect_elevation_features(self, image: np.ndarray) -> List[Dict]:
        """
        Detect specific elevation features like stairs, slopes, walls.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected elevation features
        """
        features = []
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours for building/wall detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ground_info = self.detect_ground_plane(image)
        
        for contour in contours:
            # Filter by contour size
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if this could be a vertical structure
            aspect_ratio = h / w if w > 0 else 0
            
            # Detect tall vertical structures (buildings, walls)
            if aspect_ratio > 1.2 and h > height * 0.2:
                bbox = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
                elevation = self.estimate_elevation_difference(bbox, ground_info, 'building')
                
                if elevation >= self.min_elevation_threshold:
                    features.append({
                        'type': 'elevation',
                        'class': 'vertical_structure',
                        'confidence': 0.7,
                        'bbox': bbox,
                        'elevation_difference': round(elevation, 2),
                        'feature_type': 'wall_or_building',
                        'estimated_height': round(h * 0.01, 2)  # Rough conversion
                    })
        
        # Detect stairs/steps using line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            step_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Look for short horizontal lines that could be steps
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 30:
                    step_lines.append((x1, y1, x2, y2))
            
            # Group nearby step lines
            if len(step_lines) >= 3:  # Need multiple lines for stairs
                # Simple stair detection - group lines by vertical proximity
                step_groups = []
                for line in step_lines:
                    x1, y1, x2, y2 = line
                    y_avg = (y1 + y2) / 2
                    
                    # Find if this line belongs to existing group
                    added_to_group = False
                    for group in step_groups:
                        group_y_avg = sum(l[1] + l[3] for l in group) / (2 * len(group))
                        if abs(y_avg - group_y_avg) < height * 0.1:  # Within 10% of image height
                            group.append(line)
                            added_to_group = True
                            break
                    
                    if not added_to_group:
                        step_groups.append([line])
                
                # Check for stair patterns
                for group in step_groups:
                    if len(group) >= 3:  # At least 3 step lines
                        # Calculate bounding box for the stair group
                        all_x = [coord for line in group for coord in [line[0], line[2]]]
                        all_y = [coord for line in group for coord in [line[1], line[3]]]
                        
                        bbox = {
                            'x1': min(all_x), 'y1': min(all_y),
                            'x2': max(all_x), 'y2': max(all_y)
                        }
                        
                        elevation = self.estimate_elevation_difference(bbox, ground_info, 'stairs')
                        
                        if elevation >= self.min_elevation_threshold:
                            features.append({
                                'type': 'elevation',
                                'class': 'stairs',
                                'confidence': 0.6,
                                'bbox': bbox,
                                'elevation_difference': round(elevation, 2),
                                'feature_type': 'stairs_or_steps',
                                'step_count': len(group)
                            })
        
        return features

    def _has_significant_elevation(self, elevation_diff: float) -> bool:
        """
        Check if elevation difference exceeds threshold.
        
        Args:
            elevation_diff: Elevation difference in meters
            
        Returns:
            True if elevation exceeds threshold
        """
        return elevation_diff >= self.min_elevation_threshold

    def detect_obstructions_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect obstructions using YOLO model within triangular detection area.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected obstruction objects within triangle area
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install ultralytics.")
        
        

        if isinstance(image, Image.Image):
            image = np.array(image)

        results = self.model(image)
        print('5')  
        obstructions = []
        image_height = image.shape[0]
        
        # Get ground plane information for elevation analysis
        
        ground_info = self.detect_ground_plane(image)
        
        # Process YOLO detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Check if it's an obstruction
                    obstruction_type = self._classify_obstruction(class_name)
                    if obstruction_type and confidence > 0.5:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        
                        # Estimate distance and size
                        distance = self.estimate_object_distance(bbox, class_name, image_height)
                        estimated_size = self.estimate_object_size(bbox, class_name, distance)
                        
                        # Estimate elevation difference
                        elevation_diff = self.estimate_elevation_difference(bbox, ground_info, class_name)
                        
                        # Include if larger than size threshold OR has significant elevation
                        size_check = self._is_large_enough(estimated_size)
                        elevation_check = self._has_significant_elevation(elevation_diff)
                        
                        if size_check or elevation_check:
                            obstruction_data = {
                                'type': obstruction_type,
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': bbox,
                                'area': (x2 - x1) * (y2 - y1),
                                'distance': round(distance, 2),
                                'estimated_size': estimated_size,
                                'max_dimension': max(estimated_size.values()),
                                'elevation_difference': round(elevation_diff, 2),
                                'meets_size_threshold': size_check,
                                'meets_elevation_threshold': elevation_check
                            }
                            obstructions.append(obstruction_data)
        
        # Add elevation-specific features (stairs, walls, buildings)
        elevation_features = self.detect_elevation_features(image)
        obstructions.extend(elevation_features)
        
        # Filter by triangular detection area
        filtered_obstructions = self.filter_detections_by_triangle(obstructions, image.shape)
        
        return filtered_obstructions

    def detect_obstructions_google_vision(self, image: np.ndarray) -> List[Dict]:
        """
        Detect obstructions using Google Vision API within triangular detection area.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected obstruction objects within triangle area
        """
        # Convert numpy array to bytes
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        
        # Create Vision API image object
        vision_image = self.vision_client.image_from_content(image_bytes)
        
        # Detect objects
        objects = self.vision_client.object_localization(image=vision_image).localized_object_annotations
        
        obstructions = []
        image_height = image.shape[0]
        
        # Get ground plane information for elevation analysis
        ground_info = self.detect_ground_plane(image)
        
        for obj in objects:
            obstruction_type = self._classify_obstruction(obj.name.lower())
            if obstruction_type and obj.score > 0.5:
                # Get bounding box
                vertices = obj.bounding_poly.normalized_vertices
                height, width = image.shape[:2]
                
                x1 = int(vertices[0].x * width)
                y1 = int(vertices[0].y * height)
                x2 = int(vertices[2].x * width)
                y2 = int(vertices[2].y * height)
                
                bbox = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                
                # Estimate distance and size
                distance = self.estimate_object_distance(bbox, obj.name, image_height)
                estimated_size = self.estimate_object_size(bbox, obj.name, distance)
                
                # Estimate elevation difference
                elevation_diff = self.estimate_elevation_difference(bbox, ground_info, obj.name)
                
                # Include if larger than size threshold OR has significant elevation
                size_check = self._is_large_enough(estimated_size)
                elevation_check = self._has_significant_elevation(elevation_diff)
                
                if size_check or elevation_check:
                    obstruction_data = {
                        'type': obstruction_type,
                        'class': obj.name,
                        'confidence': obj.score,
                        'bbox': bbox,
                        'area': (x2 - x1) * (y2 - y1),
                        'distance': round(distance, 2),
                        'estimated_size': estimated_size,
                        'max_dimension': max(estimated_size.values()),
                        'elevation_difference': round(elevation_diff, 2),
                        'meets_size_threshold': size_check,
                        'meets_elevation_threshold': elevation_check
                    }
                    obstructions.append(obstruction_data)
        
        # Add elevation-specific features (stairs, walls, buildings)
        elevation_features = self.detect_elevation_features(image)
        obstructions.extend(elevation_features)
        
        # Filter by triangular detection area
        filtered_obstructions = self.filter_detections_by_triangle(obstructions, image.shape)
        
        return filtered_obstructions

    def _classify_obstruction(self, class_name: str) -> Optional[str]:
        """
        Classify if a detected object is an obstruction.
        
        Args:
            class_name: Name of the detected class
            
        Returns:
            Obstruction type or None if not an obstruction
        """
        class_name_lower = class_name.lower()
        
        for obstruction_type, classes in self.obstruction_classes.items():
            if any(cls in class_name_lower for cls in classes):
                return obstruction_type
        
        return None

    def analyze_location(self, 
                        location: str, 
                        multiple_views: bool = True) -> Dict:
        """
        Analyze a location for obstructions from multiple viewpoints.
        
        Args:
            location: Address or coordinates to analyze
            multiple_views: Whether to capture multiple viewing angles
            
        Returns:
            Analysis results with detected obstructions
        """
        results = {
            'location': location,
            'total_obstructions': 0,
            'obstruction_types': {},
            'views': []
        }
        
        # Define viewing angles
        headings = [0, 90, 180, 270] if multiple_views else [0]
        
        for i, heading in enumerate(headings):
            self.logger.info(f"Processing view {i+1}/{len(headings)} (heading: {heading}°)")
            
            # Get Street View image
            image = self.get_street_view_image(location, heading=heading)
            if image is None:
                continue
            
            # Detect obstructions
            if self.model_type == "yolo":
                print('2')
                obstructions = self.detect_obstructions_yolo(image)
                
            else:
                print('3')
                obstructions = self.detect_obstructions_google_vision(image)
                print('4')
            
            # Process results
            view_result = {
                'heading': heading,
                'obstructions_count': len(obstructions),
                'obstructions': obstructions
            }
            
            results['views'].append(view_result)
            results['total_obstructions'] += len(obstructions)
            
            # Count obstruction types
            for obs in obstructions:
                obs_type = obs['type']
                results['obstruction_types'][obs_type] = results['obstruction_types'].get(obs_type, 0) + 1
            
        return results

    def visualize_obstructions(self, 
                             image: np.ndarray, 
                             obstructions: List[Dict],
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding boxes around detected obstructions.
        
        Args:
            image: Original image
            obstructions: List of detected obstructions
            save_path: Optional path to save the annotated image
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
      
        # Color map for different obstruction types
        colors = {
            'vehicle': (0, 255, 0),      # Green
            'construction': (255, 0, 0),  # Red
            'barriers': (255, 255, 0),    # Yellow
            'infrastructure': (255, 0, 255),  # Magenta
            'people': (0, 255, 255)       # Cyan
        }
        
        for obs in obstructions:
            bbox = obs['bbox']
            color = colors.get(obs['type'], (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 2)
            
            # Add label with size and elevation information
            size_info = obs.get('estimated_size', {})
            max_dim = obs.get('max_dimension', 0)
            elevation = obs.get('elevation_difference', 0)
            
            if size_info and elevation > 0:
                label = f"{obs['class']} ({obs['confidence']:.2f}) - {max_dim:.1f}m, +{elevation:.1f}m"
            elif size_info:
                label = f"{obs['class']} ({obs['confidence']:.2f}) - {max_dim:.1f}m"
            elif elevation > 0:
                label = f"{obs['class']} ({obs['confidence']:.2f}) - +{elevation:.1f}m elev"
            else:
                label = f"{obs['class']} ({obs['confidence']:.2f})"
                
            cv2.putText(annotated_image, label,
                       (bbox['x1'], bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add size and elevation details if available
            if size_info or elevation > 0:
                details = []
                if size_info:
                    details.append(f"W:{size_info.get('width', 0):.1f}m")
                    details.append(f"H:{size_info.get('height', 0):.1f}m")
                if elevation > 0:
                    details.append(f"Elev:{elevation:.1f}m")
                
                detail_text = " ".join(details)
                cv2.putText(annotated_image, detail_text,
                           (bbox['x1'], bbox['y2'] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add special marker for elevation features
            if obs.get('type') == 'elevation':
                cv2.putText(annotated_image, "ELEVATION", 
                           (bbox['x1'], bbox['y1'] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Annotated image saved to: {save_path}")
        
        return annotated_image

    def generate_report(self, analysis_results: Dict) -> str:
        """
        Generate a text report of the obstruction analysis for objects > 1.5m and elevation > 1.5m.
        
        Args:
            analysis_results: Results from analyze_location()
            
        Returns:
            Formatted report string
        """
        report = f"Obstruction & Elevation Analysis Report\n"
        report += f"{'='*70}\n"
        report += f"Location: {analysis_results['location']}\n"
        report += f"Total Obstructions Detected: {analysis_results['total_obstructions']}\n"
        report += f"Size Threshold: {self.min_size_threshold}m | Elevation Threshold: {self.min_elevation_threshold}m\n\n"
        
        # Separate obstructions by type
        size_obstructions = []
        elevation_obstructions = []
        
        for view in analysis_results['views']:
            for obs in view['obstructions']:
                if obs.get('meets_size_threshold', False):
                    size_obstructions.append(obs)
                if obs.get('meets_elevation_threshold', False) or obs.get('elevation_difference', 0) >= self.min_elevation_threshold:
                    elevation_obstructions.append(obs)
        
        report += f"Size-based Obstructions (>{self.min_size_threshold}m): {len(size_obstructions)}\n"
        report += f"Elevation-based Obstructions (>{self.min_elevation_threshold}m): {len(elevation_obstructions)}\n\n"
        
        if analysis_results['obstruction_types']:
            report += "Obstruction Types:\n"
            for obs_type, count in analysis_results['obstruction_types'].items():
                report += f"  - {obs_type.title()}: {count}\n"
        else:
            report += "No significant obstructions detected.\n"
        
        report += f"\nDetailed View Analysis:\n"
        for i, view in enumerate(analysis_results['views']):
            report += f"\nView {i+1} (Heading: {view['heading']}°): {view['obstructions_count']} obstructions\n"
            
            # Group by size vs elevation
            view_size_obs = [obs for obs in view['obstructions'] if obs.get('meets_size_threshold', False)]
            view_elev_obs = [obs for obs in view['obstructions'] if obs.get('meets_elevation_threshold', False) or obs.get('elevation_difference', 0) >= self.min_elevation_threshold]
            
            if view_size_obs:
                report += f"  Size-based obstructions: {len(view_size_obs)}\n"
                for j, obs in enumerate(view_size_obs):
                    size = obs.get('estimated_size', {})
                    max_dim = obs.get('max_dimension', 0)
                    distance = obs.get('distance', 0)
                    
                    report += f"    {j+1}. {obs['class']} - {max_dim:.1f}m at ~{distance:.1f}m\n"
            
            if view_elev_obs:
                report += f"  Elevation-based obstructions: {len(view_elev_obs)}\n"
                for j, obs in enumerate(view_elev_obs):
                    elevation = obs.get('elevation_difference', 0)
                    feature_type = obs.get('feature_type', 'unknown')
                    
                    if obs.get('type') == 'elevation':
                        report += f"    {j+1}. {obs['class']} ({feature_type}) - +{elevation:.1f}m elevation\n"
                    else:
                        report += f"    {j+1}. {obs['class']} - +{elevation:.1f}m elevation\n"
        
        return report

    def set_size_threshold(self, threshold_meters: float):
        """
        Set the minimum size threshold for obstruction detection.
        
        Args:
            threshold_meters: Minimum size in meters (default: 1.5)
        """
        self.min_size_threshold = threshold_meters
        self.logger.info(f"Size threshold set to {threshold_meters}m")

    def set_elevation_threshold(self, threshold_meters: float):
        """
        Set the minimum elevation threshold for obstruction detection.
        
        Args:
            threshold_meters: Minimum elevation difference in meters (default: 1.5)
        """
        self.min_elevation_threshold = threshold_meters
        self.logger.info(f"Elevation threshold set to {threshold_meters}m")

    def get_size_statistics(self, analysis_results: Dict) -> Dict:
        """
        Get statistics about detected obstruction sizes.
        
        Args:
            analysis_results: Results from analyze_location()
            
        Returns:
            Dictionary with size statistics
        """
        all_sizes = []
        size_by_type = {}
        
        for view in analysis_results['views']:
            for obs in view['obstructions']:
                if obs.get('meets_size_threshold', False):
                    max_dim = obs.get('max_dimension', 0)
                    all_sizes.append(max_dim)
                    
                    obs_type = obs['type']
                    if obs_type not in size_by_type:
                        size_by_type[obs_type] = []
                    size_by_type[obs_type].append(max_dim)
        
        if not all_sizes:
            return {'message': 'No large obstructions detected'}
        
        stats = {
            'total_objects': len(all_sizes),
            'average_size': round(sum(all_sizes) / len(all_sizes), 2),
            'largest_object': round(max(all_sizes), 2),
            'smallest_large_object': round(min(all_sizes), 2),
            'size_by_type': {}
        }
        
        for obs_type, sizes in size_by_type.items():
            stats['size_by_type'][obs_type] = {
                'count': len(sizes),
                'average_size': round(sum(sizes) / len(sizes), 2),
                'largest': round(max(sizes), 2)
            }
        
        return stats

    def get_elevation_statistics(self, analysis_results: Dict) -> Dict:
        """
        Get statistics about detected elevation differences.
        
        Args:
            analysis_results: Results from analyze_location()
            
        Returns:
            Dictionary with elevation statistics
        """
        all_elevations = []
        elevation_by_type = {}
        elevation_features = []
        
        for view in analysis_results['views']:
            for obs in view['obstructions']:
                elevation = obs.get('elevation_difference', 0)
                if elevation >= self.min_elevation_threshold:
                    all_elevations.append(elevation)
                    
                    obs_type = obs['type']
                    if obs_type not in elevation_by_type:
                        elevation_by_type[obs_type] = []
                    elevation_by_type[obs_type].append(elevation)
                    
                    # Track elevation-specific features
                    if obs_type == 'elevation':
                        feature_type = obs.get('feature_type', 'unknown')
                        elevation_features.append({
                            'feature': feature_type,
                            'elevation': elevation,
                            'class': obs['class']
                        })
        
        if not all_elevations:
            return {'message': 'No significant elevation differences detected'}
        
        stats = {
            'total_elevation_features': len(all_elevations),
            'average_elevation': round(sum(all_elevations) / len(all_elevations), 2),
            'highest_elevation': round(max(all_elevations), 2),
            'lowest_significant_elevation': round(min(all_elevations), 2),
            'elevation_by_type': {},
            'specific_features': elevation_features
        }
        
        for obs_type, elevations in elevation_by_type.items():
            stats['elevation_by_type'][obs_type] = {
                'count': len(elevations),
                'average_elevation': round(sum(elevations) / len(elevations), 2),
                'highest': round(max(elevations), 2)
            }
        
        return stats


# Example usage
def main():
    # Replace with your Google Street View API key
    API_KEY = 'AIzaSyB53HJkSXyesczASdc6kPv7Gv5av20rUwA' #"YOUR_GOOGLE_STREETVIEW_API_KEY"
    
    # Initialize detector (default thresholds are 1.5m for both size and elevation)
    detector = StreetViewObstructionDetector(API_KEY, model_type="yolo")
    
    # Configure detection parameters
    detector.set_size_threshold(1.5)      # Only detect objects larger than 1.5 meters
    detector.set_elevation_threshold(1.5)  # Only detect elevation differences > 1.5 meters
    
    # Set triangular detection area (forward-facing cone by default)
    detector.set_detection_triangle("forward_cone")
    
    # Example locations to analyze
    locations = [
        "Times Square, New York, NY",              # Urban area with buildings
        "Golden Gate Bridge, San Francisco, CA",   # Bridge with elevation
        "Capitol Steps, Washington, DC",           # Steps and elevation changes
        "Highway intersection, Los Angeles, CA"    # Traffic and vehicles
    ]
    
    for location in locations:
        print(f"\n{'='*70}")
        print(f"Analyzing: {location}")
        print(f"Detection area: {detector.triangle_type}")
        print(f"Size threshold: {detector.min_size_threshold}m | Elevation threshold: {detector.min_elevation_threshold}m")
        print(f"{'='*70}")
        
        # Analyze location
        results = detector.analyze_location(location, multiple_views=True)
        
        # Generate and print report
        report = detector.generate_report(results)
        print(report)
        
        # Get size statistics
        size_stats = detector.get_size_statistics(results)
        if 'message' not in size_stats:
            print(f"\nSize Statistics:")
            print(f"Total large objects in detection area: {size_stats['total_objects']}")
            print(f"Average size: {size_stats['average_size']}m")
            print(f"Largest object: {size_stats['largest_object']}m")
        
        # Get elevation statistics
        elevation_stats = detector.get_elevation_statistics(results)
        if 'message' not in elevation_stats:
            print(f"\nElevation Statistics:")
            print(f"Total elevation features in detection area: {elevation_stats['total_elevation_features']}")
            print(f"Average elevation: {elevation_stats['average_elevation']}m")
            print(f"Highest elevation: {elevation_stats['highest_elevation']}m")
            
            # Show specific elevation features
            if elevation_stats['specific_features']:
                print(f"Detected elevation features in triangle area:")
                for feature in elevation_stats['specific_features']:
                    print(f"  - {feature['feature']}: {feature['elevation']:.1f}m ({feature['class']})")
        
        # Save visualizations for first view
        if results['views']:
            first_view = results['views'][0]
            image = detector.get_street_view_image(location, heading=first_view['heading'])
            if image is not None:
                # Save detection area visualization
                area_viz = detector.visualize_detection_area_only(
                    image.copy(),
                    f"detection_area_{location.replace(' ', '_').replace(',', '')}.jpg"
                )
                
                # Save obstructions within triangle area
                obstructions_viz = detector.visualize_obstructions(
                    image.copy(), 
                    first_view['obstructions'],
                    f"triangle_detections_{location.replace(' ', '_').replace(',', '')}.jpg",
                    show_triangle_area=True
                )


def demo_triangle_types():
    """Example: Demonstrate different triangular detection areas"""
    API_KEY = "YOUR_GOOGLE_STREETVIEW_API_KEY"
    detector = StreetViewObstructionDetector(API_KEY, model_type="yolo")
    
    triangle_types = [
        "forward_cone",    # Forward-facing cone (vehicle perspective)
        "center_up",       # Triangle pointing upward from center
        "center_down",     # Triangle pointing downward from center
        "left_sector",     # Left side triangular sector
        "right_sector"     # Right side triangular sector
    ]
    
    # location = "Busy street intersection, Manhattan, NY"
    location="101 Soldiers Rd, Roleystone WA 6111, Australia"
    
    print("Triangular Detection Area Demo")
    print("="*50)
    
    for triangle_type in triangle_types:
        print(f"\nTesting triangle type: {triangle_type}")
        
        # Set triangle type
        detector.set_detection_triangle(triangle_type)
       
        # Analyze with this triangle type
        results = detector.analyze_location(location, multiple_views=False)
        print(1)
        print(f"  Objects detected in {triangle_type} area: {results['total_obstructions']}")
        
        # Save visualization
        if results['views']:
            image = detector.get_street_view_image(location)
            if image is not None:
                # Show detection area only
                area_viz = detector.visualize_detection_area_only(
                    image.copy(),
                    f"demo_{triangle_type}_area.jpg"
                )
                
                # Show detections within area
                detections_viz = detector.visualize_obstructions(
                    image.copy(),
                    results['views'][0]['obstructions'],
                    f"demo_{triangle_type}_detections.jpg",
                    show_triangle_area=True
                )
    
    print(f"\nTriangle Type Descriptions:")
    print("• forward_cone: Wide at bottom, narrow at top (vehicle forward vision)")
    print("• center_up: Triangle pointing upward from center")
    print("• center_down: Triangle pointing downward from center") 
    print("• left_sector: Left side triangular sector")
    print("• right_sector: Right side triangular sector")


def custom_triangle_demo():
    """Example: Create custom triangular detection area"""
    API_KEY = "YOUR_GOOGLE_STREETVIEW_API_KEY"
    detector = StreetViewObstructionDetector(API_KEY, model_type="yolo")
    
    # Define custom triangle points (as fractions of image dimensions)
    # These will be scaled to actual image size during detection
    custom_points = [
        (320, 100),  # Top center
        (100, 500),  # Bottom left  
        (540, 500)   # Bottom right
    ]
    
    # Set custom triangle
    detector.set_detection_triangle("custom", custom_points)
    
    location = "Public square with monument, Washington DC"
    results = detector.analyze_location(location)
    
    print("Custom Triangle Detection Demo")
    print("="*40)
    print(f"Custom triangle points: {custom_points}")
    print(f"Objects detected in custom area: {results['total_obstructions']}")
    
    if results['views']:
        image = detector.get_street_view_image(location)
        if image is not None:
            # Visualize custom triangle area
            custom_viz = detector.visualize_obstructions(
                image,
                results['views'][0]['obstructions'],
                "custom_triangle_demo.jpg",
                show_triangle_area=True
            )


def compare_detection_areas():
    """Compare detections across different triangle areas"""
    API_KEY = "YOUR_GOOGLE_STREETVIEW_API_KEY"
    detector = StreetViewObstructionDetector(API_KEY, model_type="yolo")
    
    location = "Times Square, New York, NY"
    triangle_types = ["forward_cone", "center_up", "left_sector", "right_sector"]
    
    print("Detection Area Comparison")
    print("="*40)
    print(f"Location: {location}")
    
    comparison_results = {}
    
    for triangle_type in triangle_types:
        detector.set_detection_triangle(triangle_type)
        results = detector.analyze_location(location, multiple_views=False)
        
        comparison_results[triangle_type] = {
            'total_objects': results['total_obstructions'],
            'by_type': results['obstruction_types']
        }
        
        print(f"\n{triangle_type.upper()}:")
        print(f"  Total objects: {results['total_obstructions']}")
        if results['obstruction_types']:
            for obs_type, count in results['obstruction_types'].items():
                print(f"  - {obs_type}: {count}")
    
    # Find best triangle type for this location
    best_triangle = max(comparison_results.keys(), 
                       key=lambda x: comparison_results[x]['total_objects'])
    
    print(f"\nMost effective detection area: {best_triangle}")
    print(f"Detected {comparison_results[best_triangle]['total_objects']} objects")


if __name__ == "__main__":
    # main()
    
    # Uncomment to test different triangle configurations
    demo_triangle_types()
    # custom_triangle_demo()
    # compare_detection_areas()
