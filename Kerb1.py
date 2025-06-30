import requests
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PolygonElevationAnalyzer:
    def __init__(self, google_api_key):
        self.api_key = google_api_key
        
    def sample_points_in_polygon(self, vertices, num_points=20):
        """Generate evenly spaced points within a triangle"""
        # Create barycentric coordinate combinations
        points = []
        for i in range(num_points):
            for j in range(num_points - i):
                u = np.random.rand()
                v = np.random.rand() * (1 - u)
                w = 1 - u - v
                # Convert to Cartesian coordinates
                x = u*vertices[0][0] + v*vertices[1][0] + w*vertices[2][0]
                y = u*vertices[0][1] + v*vertices[1][1] + w*vertices[2][1]
                points.append((x, y))
        return points
    
    def get_bulk_elevations(self, points):
        """Get elevations for multiple points in a single request"""
        locations = "|".join([f"{lat},{lng}" for lat, lng in points])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={self.api_key}"
        response = requests.get(url).json()
        return [result['elevation'] for result in response.get('results', [])]
    
    def analyze_polygon_elevation(self, driveway_coords, kerb_coords, right_coords):
        """Calculate max elevation within the polygon"""
        # Convert to (lat, lng) tuples
        vertices = [
            (driveway_coords[0], driveway_coords[1]),
            (kerb_coords[0], kerb_coords[1]),
            (right_coords[0], right_coords[1])
        ]
        
        # Sample points within polygon
        sample_points = self.sample_points_in_polygon(vertices)
        elevations = self.get_bulk_elevations(sample_points)
        
        if not elevations:
            raise Exception("Failed to get elevation data")
        
        # Find max elevation
        max_elevation = max(elevations)
        max_point = sample_points[elevations.index(max_elevation)]
        
        # Create elevation grid for visualization
        grid_x, grid_y = np.mgrid[
            min(v[0] for v in vertices):max(v[0] for v in vertices):100j,
            min(v[1] for v in vertices):max(v[1] for v in vertices):100j
        ]
        points = np.array([(p[0], p[1]) for p in sample_points])
        values = np.array(elevations)
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        # Visualization
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='terrain', alpha=0.7)
        ax.scatter(*zip(*sample_points), elevations, c='r', s=10)
        ax.scatter(max_point[0], max_point[1], max_elevation, c='b', s=100, marker='*')
        
        # Plot polygon edges
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i+1)%len(vertices)]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   [0, 0], 'k-', linewidth=2)
        
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Elevation (m)')
        plt.title("Elevation Analysis Within Polygon")
        plt.colorbar(surf, shrink=0.5, aspect=5, label='Elevation (m)')
        
        # Add annotation for max point
        ax.text(max_point[0], max_point[1], max_elevation, 
                f"Max: {max_elevation:.2f}m", color='blue')
        
        plt.tight_layout()
        plt.show()
        
        return {
            "max_elevation": max_elevation,
            "max_point": max_point,
            "sample_points": sample_points,
            "elevations": elevations,
            "visualization": fig
        }

# Example usage with previous results
if __name__ == "__main__":
    # Assuming we have previous analysis results
    previous_results = {
        "driveway_coords": (-32.040512, 115.993215),
        "kerb_coords": (-32.040528, 115.993230),
        "right_coords": (-32.040510, 115.993270)
    }
    
    analyzer = PolygonElevationAnalyzer("YOUR_GOOGLE_API_KEY")
    elevation_results = analyzer.analyze_polygon_elevation(
        previous_results["driveway_coords"],
        previous_results["kerb_coords"],
        previous_results["right_coords"]
    )
    
    print(f"\nMaximum elevation within polygon: {elevation_results['max_elevation']:.2f} meters")
    print(f"Location of max elevation: {elevation_results['max_point'][0]:.6f}, {elevation_results['max_point'][1]:.6f}")
