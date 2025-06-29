import cv2
import numpy as np
import requests
from ultralytics import YOLO
from geopy.distance import geodesic
import matplotlib.pyplot as plt

# --- Step 1: Fetch Google Maps Image ---
def get_google_maps_image(lat, lon, zoom=20, size="640x640", api_key="YOUR_API_KEY"):
    """Download satellite image from Google Maps Static API"""
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={api_key}"
    response = requests.get(url)
    with open("road_image.png", "wb") as f:
        f.write(response.content)
    return cv2.imread("road_image.png")

# --- Step 2: AI Road Segmentation ---
def detect_road_mask(image):
    model = YOLO("yolov8n-seg.pt")  # Pre-trained (fine-tune for roads)
    results = model(image)
    return results[0].masks[0].data.cpu().numpy().squeeze()  # Largest mask

# --- Step 3: Calculate Road Width ---
def calculate_width(mask, lat, lon, zoom):
    # Find edges
    edges = cv2.Canny((mask*255).astype(np.uint8), 50, 150)
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Calculate ground resolution (meters/pixel)
    # At zoom 20, 1px ≈ 0.1m (varies by latitude)
    mp_per_px = 156543.03392 * np.cos(lat * np.pi/180) / (2 ** zoom)
    
    # Measure perpendicular distances
    if len(edge_points) > 1:
        left_edge = edge_points[edge_points[:,1].argmin()]
        right_edge = edge_points[edge_points[:,1].argmax()]
        width_px = abs(right_edge[1] - left_edge[1])
        return width_px * mp_per_px
    return 0

# --- Execution ---
if __name__ == "__main__":
    # Example: Hay Street, Perth
    lat, lon = -31.9559, 115.8606  # Coordinates
    zoom = 20  # 19-21 for best results
    
    # Get image
    img = get_google_maps_image(lat, lon, zoom)
    
    # Detect road
    mask = detect_road_mask(img)
    
    # Calculate width
    width = calculate_width(mask, lat, lon, zoom)
    print(f"Estimated road width: {width:.1f} meters")

    # Visualize
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, alpha=0.3, cmap='jet')
    plt.title(f"Road Width: {width:.1f}m")
    plt.show()
