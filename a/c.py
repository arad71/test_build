import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.distance import distance
import math
import requests

# -------- CONFIG --------
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # <<<<<< Replace this

# -------- UTILS --------
def move_point(origin, angle_deg, dist_m):
    """Returns new lat/lon moved from origin in angle and meters."""
    return distance(meters=dist_m).destination(origin, angle_deg)

def calculate_bearing(p1, p2):
    """Calculates compass bearing from p1 to p2 (lat/lon)."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def get_snapped_point_and_bearing(lat, lon):
    """Uses Google Roads API to snap and get road direction bearing."""
    path = f"{lat},{lon}|{lat + 0.0001},{lon}"
    url = (
        "https://roads.googleapis.com/v1/snapToRoads"
        f"?path={path}&interpolate=true&key={GOOGLE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "snappedPoints" in data and len(data["snappedPoints"]) >= 2:
        pt1 = data["snappedPoints"][0]["location"]
        pt2 = data["snappedPoints"][1]["location"]
        snapped_point = (pt1["latitude"], pt1["longitude"])
        bearing = calculate_bearing((pt1["latitude"], pt1["longitude"]),
                                    (pt2["latitude"], pt2["longitude"]))
        return snapped_point, bearing
    else:
        return None, None

# -------- STREAMLIT UI --------
st.title("Click to Draw Triangle Aligned with Google Road (Snapped + Arrow)")

# Map setup
m = folium.Map(location=[-31.95, 115.86], zoom_start=19)
m.add_child(folium.LatLngPopup())
clicked_data = st_folium(m, width=700, height=500)

# -------- HANDLE CLICK --------
if clicked_data and clicked_data['last_clicked']:
    lat = clicked_data['last_clicked']['lat']
    lon = clicked_data['last_clicked']['lng']
    clicked_point = (lat, lon)

    # SNAP to road and get bearing
    snapped_point, road_bearing = get_snapped_point_and_bearing(lat, lon)

    if snapped_point and road_bearing is not None:
        # Triangle arms (from snapped point)
        pt1 = move_point(snapped_point, road_bearing + 45, 5)
        pt2 = move_point(snapped_point, road_bearing - 60, 5)
        triangle = [snapped_point, (pt1.latitude, pt1.longitude), (pt2.latitude, pt2.longitude)]

        # Draw arrow in road direction
        arrow_end = move_point(snapped_point, road_bearing, 6)

        # Redraw map
        m = folium.Map(location=snapped_point, zoom_start=19)

        folium.Marker(location=clicked_point, tooltip="Original Click").add_to(m)
        folium.Marker(location=snapped_point, tooltip="Snapped to Road", icon=folium.Icon(color="green")).add_to(m)

        # Triangle
        folium.Polygon(
            locations=triangle,
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

        # Arrow (road direction)
        folium.PolyLine(
            locations=[snapped_point, (arrow_end.latitude, arrow_end.longitude)],
            color='red',
            weight=3,
            opacity=0.9
        ).add_to(m)

        # Show final map
        st_folium(m, width=700, height=500)
    else:
        st.error("Google Roads API could not snap to road or get direction.")
