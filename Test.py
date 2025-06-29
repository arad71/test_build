import cv2
import numpy as np
import rasterio
from ultralytics import YOLO
from geopy.distance import geodesic
import geopandas as gpd
import matplotlib.pyplot as plt

# --- Step 1: Load Road Detection AI Model ---
model = YOLO("yolov8n-seg.pt")  # Pre-trained (fine-tune for roads)

# --- Step 2: Detect Road Edges in Image ---
def detect_road_edges(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    # Get largest road mask (assuming road is largest segmented object)
    road_mask = np.zeros_like(img[:,:,0])
    for mask in results[0].masks:
        if mask.data.shape[0] > np.sum(road_mask):  # Compare area
            road_mask = mask.data.cpu().numpy().squeeze().astype(np.uint8)
    
    return road_mask

# --- Step 3: Calculate Road Width ---
def calculate_road_width(mask, ground_resolution=0.1):
    """
    Args:
        mask: Binary road mask (1=road, 0=non-road)
        ground_resolution: meters/pixel (e.g., 0.1m for drone imagery)
    Returns:
        Average width in meters
    """
    # Find edges using Canny
    edges = cv2.Canny(mask*255, 50, 150)
    
    # Get all edge points
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Calculate pairwise distances between edge points
    from scipy.spatial import cKDTree
    tree = cKDTree(edge_points)
    distances, _ = tree.query(edge_points, k=2)  # Distance to nearest neighbor
    
    # Filter plausible widths (5-50m typical roads)
    valid_distances = distances[(distances[:,1] > 5) & (distances[:,1] < 50)]
    avg_width_pixels = np.median(valid_distances[:,1])
    
    return avg_width_pixels * ground_resolution

# --- Step 4: Geospatial Integration ---
def get_ground_resolution(bbox):
    """Calculate meters/pixel from image bounds (bbox = [min_lon, min_lat, max_lon, max_lat])"""
    width_meters = geodesic((bbox[1], bbox[0]), (bbox[1], bbox[2])).meters
    return width_meters / 1024  # Assuming 1024px image width

# --- Execution ---
if __name__ == "__main__":
    # Example for Perth road
    image_path = "perth_road.jpg"
    bbox = [115.85, -31.95, 115.86, -31.94]  # Image bounding box
    
    # Detect road
    road_mask = detect_road_edges(image_path)
    
    # Calculate width
    resolution = get_ground_resolution(bbox)
    width = calculate_road_width(road_mask, resolution)
    
    print(f"Estimated road width: {width:.1f} meters")
    
    # Visualize
    plt.imshow(road_mask, cmap='gray')
    plt.title(f"Road Width: {width:.1f}m")
    plt.show()
