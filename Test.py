import requests
import math
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json

class IntersectionFinder:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="intersection_finder")
    
    def get_coordinates(self, address):
        """Get latitude and longitude from address"""
        try:
            location = self.geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            print(f"Error geocoding address: {e}")
            return None
    
    def haversine_distance(self, coord1, coord2):
        """Calculate distance between two coordinates in kilometers"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def find_intersections_overpass(self, lat, lon, radius=500):
        """Find intersections using OpenStreetMap Overpass API"""
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Query to find highway intersections within radius
        overpass_query = f"""
        [out:json];
        (
          node
            ["highway"~"^(traffic_signals|crossing|stop)$"]
            (around:{radius},{lat},{lon});
          node
            (way["highway"]["highway"!~"footway|cycleway|path"]
             (around:{radius},{lat},{lon});)
            (way["highway"]["highway"!~"footway|cycleway|path"]
             (around:{radius},{lat},{lon}););
        );
        out geom;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            data = response.json()
            
            intersections = []
            for element in data.get('elements', []):
                if element['type'] == 'node' and 'lat' in element and 'lon' in element:
                    intersection_coord = (element['lat'], element['lon'])
                    distance = self.haversine_distance((lat, lon), intersection_coord)
                    
                    intersections.append({
                        'coordinates': intersection_coord,
                        'distance_km': distance,
                        'distance_m': distance * 1000,
                        'tags': element.get('tags', {})
                    })
            
            # Sort by distance
            intersections.sort(key=lambda x: x['distance_km'])
            return intersections
            
        except Exception as e:
            print(f"Error querying Overpass API: {e}")
            return []
    
    def find_intersections_nominatim(self, lat, lon, radius=0.01):
        """Find nearby streets using Nominatim reverse geocoding"""
        try:
            # Get multiple nearby locations
            nearby_locations = []
            
            # Sample points around the address
            offsets = [
                (0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius),
                (radius/2, radius/2), (-radius/2, -radius/2), 
                (radius/2, -radius/2), (-radius/2, radius/2)
            ]
            
            for lat_offset, lon_offset in offsets:
                try:
                    location = self.geolocator.reverse((lat + lat_offset, lon + lon_offset))
                    if location and location.raw.get('display_name'):
                        address_parts = location.raw.get('display_name', '').split(', ')
                        if len(address_parts) >= 2:
                            nearby_locations.append({
                                'coordinates': (lat + lat_offset, lon + lon_offset),
                                'address': location.raw.get('display_name'),
                                'distance_km': self.haversine_distance((lat, lon), (lat + lat_offset, lon + lon_offset))
                            })
                except:
                    continue
            
            # Sort by distance
            nearby_locations.sort(key=lambda x: x['distance_km'])
            return nearby_locations[:5]  # Return top 5
            
        except Exception as e:
            print(f"Error with Nominatim search: {e}")
            return []
    
    def find_nearest_intersection(self, address, method='overpass'):
        """Main method to find nearest intersection"""
        print(f"Finding intersection for: {address}")
        
        # Get coordinates of the address
        coords = self.get_coordinates(address)
        if not coords:
            return None
        
        lat, lon = coords
        print(f"Address coordinates: {lat:.6f}, {lon:.6f}")
        
        if method == 'overpass':
            intersections = self.find_intersections_overpass(lat, lon)
        else:
            intersections = self.find_intersections_nominatim(lat, lon)
        
        if not intersections:
            print("No intersections found nearby")
            return None
        
        return {
            'original_address': address,
            'original_coordinates': coords,
            'nearest_intersections': intersections[:3],  # Return top 3
            'method_used': method
        }

# Alternative simpler version using just geopy
def simple_intersection_finder(address):
    """Simplified version using geopy only"""
    geolocator = Nominatim(user_agent="simple_intersection_finder")
    
    try:
        # Get the main location
        location = geolocator.geocode(address)
        if not location:
            return None
        
        lat, lon = location.latitude, location.longitude
        
        # Try to find cross streets by looking at nearby addresses
        cross_streets = []
        
        # Get reverse geocoding information which often includes cross streets
        reverse_location = geolocator.reverse((lat, lon), exactly_one=False, limit=5)
        
        for loc in reverse_location:
            if loc.raw.get('display_name'):
                cross_streets.append({
                    'description': loc.raw.get('display_name'),
                    'coordinates': (loc.latitude, loc.longitude),
                    'distance_m': geodesic((lat, lon), (loc.latitude, loc.longitude)).meters
                })
        
        return {
            'address': address,
            'coordinates': (lat, lon),
            'nearby_locations': cross_streets
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Initialize the finder
    finder = IntersectionFinder()
    
    # Example address
    test_address = "1600 Pennsylvania Avenue NW, Washington, DC"
    
    print("=" * 50)
    print("METHOD 1: Using Overpass API (OpenStreetMap)")
    print("=" * 50)
    
    result = finder.find_nearest_intersection(test_address, method='overpass')
    if result:
        print(f"\nOriginal Address: {result['original_address']}")
        print(f"Coordinates: {result['original_coordinates']}")
        print("\nNearest Intersections:")
        for i, intersection in enumerate(result['nearest_intersections'], 1):
            print(f"{i}. Distance: {intersection['distance_m']:.1f}m")
            print(f"   Coordinates: {intersection['coordinates']}")
            print(f"   Tags: {intersection['tags']}")
            print()
    
    print("=" * 50)
    print("METHOD 2: Using Nominatim (Simpler)")
    print("=" * 50)
    
    result2 = finder.find_nearest_intersection(test_address, method='nominatim')
    if result2:
        print(f"\nOriginal Address: {result2['original_address']}")
        print(f"Coordinates: {result2['original_coordinates']}")
        print("\nNearby Locations:")
        for i, location in enumerate(result2['nearest_intersections'], 1):
            print(f"{i}. Distance: {location['distance_km']*1000:.1f}m")
            print(f"   Address: {location['address']}")
            print()
    
    print("=" * 50)
    print("METHOD 3: Simple Version")
    print("=" * 50)
    
    simple_result = simple_intersection_finder(test_address)
    if simple_result:
        print(f"\nAddress: {simple_result['address']}")
        print(f"Coordinates: {simple_result['coordinates']}")
        print("\nNearby Locations:")
        for i, loc in enumerate(simple_result['nearby_locations'], 1):
            print(f"{i}. {loc['description']}")
            print(f"   Distance: {loc['distance_m']:.1f}m")
            print()
