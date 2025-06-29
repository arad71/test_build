# Requirements and Setup Instructions

## requirements.txt
```
opencv-python==4.8.1.78
numpy==1.24.3
requests==2.31.0
Pillow==10.0.0
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.181
google-generativeai==0.1.0
google-cloud-vision==3.4.4
folium==0.14.0
geopy==2.3.0
sqlite3
logging
dataclasses
typing
json
math
os
datetime
```

## Installation Steps

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Google APIs

#### Google Maps API:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Maps JavaScript API
   - Street View Static API
   - Geocoding API
4. Create credentials (API Key)
5. Restrict the API key to your specific APIs

#### Google Cloud Vision API (Optional):
1. Enable Cloud Vision API in Google Cloud Console
2. Create a service account
3. Download the JSON credentials file
4. Set the environment variable or provide path in code

### 3. Download YOLO Model
The code will automatically download YOLOv8 model on first run, but you can also:
```bash
# Download specific YOLO models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 4. Configuration
Update the main() function with your API keys:
```python
GOOGLE_MAPS_API_KEY = "your_actual_api_key_here"
GOOGLE_VISION_CREDENTIALS = "path/to/your/credentials.json"
```

## Usage Examples

### Basic Usage
```python
from infrastructure_mapper import InfrastructureMapper

# Initialize
mapper = InfrastructureMapper("your_api_key")

# Process an address
detections = mapper.process_address("123 Main St, Anytown, USA")

# Process coordinates
detections = mapper.process_location(40.7128, -74.0060)

# Create map
map_obj = mapper.create_area_map(40.7128, -74.0060)
map_obj.save("detections.html")
```

### Advanced Usage
```python
# Batch process area
detections = mapper.batch_process_area(
    center_lat=40.7128, 
    center_lng=-74.0060,
    grid_spacing=0.001
)

# Get detections from database
saved_detections = mapper.data_manager.get_detections_in_area(
    40.7128, -74.0060, radius_km=2.0
)
```

## API Costs Estimation

### Google Maps API Pricing (as of 2024):
- **Street View Static API**: $7 per 1,000 requests
- **Geocoding API**: $5 per 1,000 requests
- **Maps JavaScript API**: $7 per 1,000 map loads

### Google Cloud Vision API:
- **Object Localization**: $1.50 per 1,000 images

### Example Cost Calculation:
For processing 100 locations with 4 angles each:
- 400 Street View requests: $2.80
- 400 Vision API requests: $0.60
- **Total: ~$3.40**

## Performance Optimization

### Caching Strategy:
```python
# Enable result caching
mapper.data_manager.cache_enabled = True

# Set cache expiry (days)
mapper.data_manager.cache_expiry = 30
```

### Batch Processing:
```python
# Process multiple locations efficiently
locations = [(lat1, lng1), (lat2, lng2), (lat3, lng3)]
for lat, lng in locations:
    detections = mapper.process_location(lat, lng, headings=[0, 180])
```

### Memory Management:
```python
# Clear model cache periodically
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

## Troubleshooting

### Common Issues:

1. **API Key Errors**:
   - Verify API key is correct
   - Check that required APIs are enabled
   - Ensure billing is enabled on Google Cloud

2. **Model Loading Issues**:
   - Check internet connection for YOLO download
   - Ensure sufficient disk space
   - Try different YOLO model sizes (yolov8n.pt is smallest)

3. **Coordinate Accuracy**:
   - Objects closer to camera are more accurate
   - Multiple viewing angles improve accuracy
   - Consider ground truth validation for critical applications

4. **Performance Issues**:
   - Use GPU for YOLO inference if available
   - Reduce image resolution for faster processing
   - Implement parallel processing for batch operations

### Debug Mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
```

## Extending the System

### Adding New Object Types:
```python
# In ObjectDetector class
self.infrastructure_classes['new_object'] = ['synonym1', 'synonym2']

# Add size estimation
object_sizes = {
    'new_object': 2.5  # meters
}
```

### Custom Detection Models:
```python
# Train custom model for specific objects
def load_custom_model(self, model_path):
    self.custom_models['fence'] = torch.load(model_path)

# Use in detection pipeline
custom_detections = self._detect_with_custom_model(image, 'fence')
```

### Integration with Other APIs:
```python
# Add satellite imagery
def get_satellite_image(self, lat, lng):
    # Use Google Maps Static API or other satellite providers
    pass

# Add real-time traffic data
def get_traffic_conditions(self, lat, lng):
    # Use Google Maps Traffic API
    pass
```
