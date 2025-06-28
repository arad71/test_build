import geopandas as gpd
import requests

# Kalamunda bounding box (approximate)
bbox = "116.00,-32.00,116.20,-31.90"  # Adjust as needed
url = f"https://data.landgate.wa.gov.au/api/geospatial/cadastre?format=geojson&bbox={bbox}"
response = requests.get(url)

with open("kalamunda_lots.geojson", "wb") as f:
    f.write(response.content)

# Load into GeoPandas
gdf = gpd.read_file("kalamunda_lots.geojson")
print(gdf.head())  # Check data
