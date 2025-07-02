import requests
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from math import radians, cos, sin, asin, sqrt

# Configuration
GOOGLE_API_KEY = "YOUR_API_KEY"
MAX_DISTANCE = 10  # meters
OBSTRUCTION_CLASSES = ['tree', 'fence', 'wall', 'vehicle', 'building', 'pole']

# Load pre-trained DETR model for object detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def get_streetview_panorama(lat, lon, heading, pitch=0, fov=120):
    """Get high-quality Street View panorama"""
    params = {
        'size': '1024x1024',  # higher resolution
        'location': f'{lat},{lon}',
        'heading': heading,
        'pitch': pitch,
        'fov': fov,  # wider field of view
        'source': 'outdoor',
        'key': GOOGLE_API_KEY
    }
    url = 'https://maps.googleapis.com/maps/api/streetview?'
    response = requests.get(url, params=params)
    return Image.open(requests.get(response.url, stream=True).raw)

def detect_obstructions_ml(image):
    """Use ML model to detect specific obstruction types"""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )[0]
    
    obstructions = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = model.config.id2label[label.item()]
        if class_name.lower() in OBSTRUCTION_CLASSES:
            obstructions.append({
                "class": class_name,
                "confidence": round(score.item(), 3),
                "box": [round(i, 2) for i in box.tolist()]
            })
    
    return obstructions

def estimate_distance_to_obstruction(image, obstruction, heading, fov):
    """
    Estimate distance to obstruction based on its size in image
    and known Street View camera parameters
    """
    # Get bounding box dimensions
    _, _, width, height = obstruction['box']
    
    # Simplified distance estimation (would need calibration)
    # Larger objects in center of image are assumed closer
    area = width * height
    if area > 0.3 * image.size[0] * image.size[1]:  # occupies >30% of image
        return MAX_DISTANCE * 0.3  # very close
    elif area > 0.1 * image.size[0] * image.size[1]:
        return MAX_DISTANCE * 0.6
    else:
        return MAX_DISTANCE * 0.9

def analyze_driveway_visibility_ml(driveway_coords, road_coords):
    """
    Enhanced visibility analysis using ML models
    Returns obstruction report
    """
    # Calculate heading from driveway to road
    heading = np.arctan2(road_coords['lon'] - driveway_coords['lon'], 
                         road_coords['lat'] - driveway_coords['lat']) * 180 / np.pi
    
    # Get panorama image
    panorama = get_streetview_panorama(
        driveway_coords['lat'], 
        driveway_coords['lon'], 
        heading,
        fov=120  # wider view
    )
    
    # Detect obstructions with ML
    obstructions = detect_obstructions_ml(panorama)
    
    # Analyze results
    report = {
        'location': driveway_coords,
        'heading': heading,
        'obstructions': [],
        'clear_view': True
    }
    
    for obs in obstructions:
        distance = estimate_distance_to_obstruction(panorama, obs, heading, fov=120)
        if distance <= MAX_DISTANCE:
            report['clear_view'] = False
            report['obstructions'].append({
                'type': obs['class'],
                'confidence': obs['confidence'],
                'estimated_distance_meters': round(distance, 1)
            })
    
    return report

# Example usage
if __name__ == "__main__":
    # Example coordinates (replace with actual)
    driveway = {'lat': 37.4220656, 'lon': -122.0840897}
    road = {'lat': 37.4220656, 'lon': -122.0840897 + 0.0001}
    
    report = analyze_driveway_visibility_ml(driveway, road)
    
    print("\nDriveway Visibility Analysis Report:")
    print(f"Location: {report['location']}")
    print(f"View Direction: {round(report['heading'], 1)}°")
    
    if report['clear_view']:
        print("✅ Visibility is clear within 10m of the driveway")
    else:
        print("⚠️ Potential vision obstructions detected:")
        for obs in report['obstructions']:
            print(f" - {obs['type']} ({obs['confidence']*100}% confidence) ~{obs['estimated_distance_meters']}m away")
