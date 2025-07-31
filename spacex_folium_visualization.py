
#  SpaceX Launch Site Interactive Map Visualization


# Required installations (for JupyterLite environments)
import piplite
await piplite.install(['folium', 'pandas'])

# Imports
import folium
import pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
from js import fetch
import io
from math import radians, sin, cos, sqrt, atan2


#  Load Data

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
response = await fetch(URL)
spacex_csv = io.BytesIO((await response.arrayBuffer()).to_py())
spacex_df = pd.read_csv(spacex_csv)

# Clean and select relevant columns
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]

# Create base map centered at mean lat/long
launch_map = folium.Map(location=[spacex_df['Lat'].mean(), spacex_df['Long'].mean()], zoom_start=5)


#  Mark all Launch Sites

launch_sites_df = spacex_df.groupby('Launch Site', as_index=False).first()
for _, row in launch_sites_df.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=row['Launch Site'],
        icon=folium.Icon(color='blue', icon='rocket', prefix='fa')
    ).add_to(launch_map)

# Optional: Save to HTML
launch_map.save("Launch_Sites_Map.html")


#  Mark NASA Johnson Space Center

nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))

marker = folium.map.Marker(
    nasa_coordinate,
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>NASA JSC</b></div>'
    )
)

site_map.add_child(circle)
site_map.add_child(marker)


#  Mark Success (1) and Failed (0) Launches

launch_map = folium.Map(location=[spacex_df['Lat'].mean(), spacex_df['Long'].mean()], zoom_start=5)

for _, row in spacex_df.iterrows():
    color = 'green' if row['class'] == 1 else 'red'
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=f"Site: {row['Launch Site']}<br>Outcome: {'Success' if row['class'] == 1 else 'Failure'}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(launch_map)

# Add Mouse Position plugin
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='bottomright',
    separator=' | ',
    empty_string='Unavailable',
    lng_first=True,
    num_digits=5,
    prefix='Coordinates:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)
launch_map.add_child(mouse_position)


#  Extension: Marker Clusters

marker_cluster = MarkerCluster()
site_map.add_child(marker_cluster)

for _, row in spacex_df.iterrows():
    color = 'green' if row['class'] == 1 else 'red'
    marker = folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=f"{row['Launch Site']} - {'Success' if row['class'] == 1 else 'Failure'}",
        icon=folium.Icon(color=color)
    )
    marker_cluster.add_child(marker)

#  Distance Calculations Using Haversine Formula


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Reference location: Cocoa Beach
ref_lat, ref_lon = 28.3200, -80.6076

spacex_df['distance_to_coast_km'] = spacex_df.apply(
    lambda row: calculate_distance(row['Lat'], row['Long'], ref_lat, ref_lon),
    axis=1
)


#  Task 4: Visualize Distance on Map


launch_site = (28.562302, -80.577356)
coastline = (28.56367, -80.57163)

distance_km = calculate_distance(*launch_site, *coastline)

distance_map = folium.Map(location=launch_site, zoom_start=15)

# Markers
folium.Marker(launch_site, popup='Launch Site', icon=folium.Icon(color='blue')).add_to(distance_map)
folium.Marker(coastline, popup='Coastline Point', icon=folium.Icon(color='green')).add_to(distance_map)

# Line
folium.PolyLine([launch_site, coastline], color='orange', weight=2).add_to(distance_map)

# Distance Label
midpoint = [(launch_site[0] + coastline[0]) / 2, (launch_site[1] + coastline[1]) / 2]
folium.Marker(
    location=midpoint,
    icon=DivIcon(
        icon_size=(150, 36),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12px; color:#d35400;"><b>{distance_km:.2f} KM</b></div>',
    )
).add_to(distance_map)


#  Bonus: Distance to Cocoa Beach


cocoa_beach = (28.3556, -80.6110)
city_distance_km = calculate_distance(*launch_site, *cocoa_beach)

city_map = folium.Map(location=launch_site, zoom_start=10)
folium.Marker(launch_site, popup='Launch Site', icon=folium.Icon(color='blue')).add_to(city_map)
folium.Marker(cocoa_beach, popup='Cocoa Beach', icon=folium.Icon(color='purple')).add_to(city_map)
folium.PolyLine([launch_site, cocoa_beach], color='purple', weight=2).add_to(city_map)

city_midpoint = [(launch_site[0] + cocoa_beach[0]) / 2, (launch_site[1] + cocoa_beach[1]) / 2]
folium.Marker(
    location=city_midpoint,
    icon=DivIcon(
        icon_size=(150, 36),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12px; color:#8e44ad;"><b>{city_distance_km:.2f} KM</b></div>',
    )
).add_to(city_map)
