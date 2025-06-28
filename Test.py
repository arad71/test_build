import geopandas as gpd
import requests

# Download WA cadastre data (example for Perth suburbs)
url = "https://data.landgate.wa.gov.au/api/geospatial/cadastre?format=geojson&bbox=115.80,-32.05,116.00,-31.90"  # Bounding box for Perth
response = requests.get(url)
with open("wa_lots.geojson", "wb") as f:
    f.write(response.content)

# Load into GeoPandas
gdf = gpd.read_file("wa_lots.geojson")
print(gdf.head())  # Print first 5 lot boundaries
