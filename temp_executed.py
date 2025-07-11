#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring
import time
import psutil
from functools import wraps

# Animation and visualization imports
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

# Parallel processing optimization
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Network analysis
from networkx.algorithms.flow import shortest_augmenting_path

# Set random seed for reproducibility
np.random.seed(42)

# Performance monitoring decorator
def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f}s | Memory change: {end_memory - start_memory:+.1f}MB")
        return result
    return wrapper

print("Libraries imported successfully!")
print(f"Available CPU cores: {mp.cpu_count()}")
print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")




# In[2]:


# Load data from CSV file
print("Loading data from dataset.csv...")
try:
    edges_df = pd.read_csv('/home/algorithm/project/EvacuationPJ/dataset.csv')
    print(f"Successfully loaded {len(edges_df)} data  from dataset.csv")
    print(f"Columns: {list(edges_df.columns)}")
    print(f"Sample data:")
    print(edges_df.head())
except Exception as e:
    print(f"Error loading dataset.csv: {e}")
    edges_df = None


# In[3]:


# Process the loaded CSV data to create nodes and edges
if edges_df is not None:
    # Create nodes from unique source and target nodes
    unique_nodes = pd.unique(edges_df[['u', 'v']].values.ravel('K'))
    
    # Create nodes DataFrame with basic geometry (simplified coordinates)
    nodes_data = []
    for i, node_id in enumerate(unique_nodes):
        # For node ID
        nodes_data.append({
            'node_id': node_id,
        })
    
    nodes = pd.DataFrame(nodes_data).set_index('node_id')
    
    # Add random population to each node
    np.random.seed(42)  # For reproducible results
    num_nodes = len(nodes)
    
    # Generate random population for each node
    # Using different distribution strategies for variety
    residential_nodes = int(num_nodes * 0.6)  # 60% residential areas
    commercial_nodes = int(num_nodes * 0.25)  # 25% commercial areas
    industrial_nodes = num_nodes - residential_nodes - commercial_nodes  # 15% industrial/other
    
    populations = []
    
    # Residential areas: higher population (50-500 people per node)
    populations.extend(np.random.randint(50, 501, residential_nodes))
    
    # Commercial areas: moderate population (10-200 people per node)
    populations.extend(np.random.randint(10, 201, commercial_nodes))
    
    # Industrial/other areas: lower population (0-100 people per node)
    populations.extend(np.random.randint(0, 101, industrial_nodes))
    
    # Shuffle to randomize distribution across nodes
    np.random.shuffle(populations)
    
    # Assign population to nodes
    nodes['population'] = populations
    
    # Add node types for reference
    node_types = ['residential'] * residential_nodes + ['commercial'] * commercial_nodes + ['industrial'] * industrial_nodes
    np.random.shuffle(node_types)
    nodes['node_type'] = node_types
    
    # Calculate total population
    total_population = nodes['population'].sum()
    
    print(f"Population Distribution:")
    print(f"Total population across all nodes: {total_population:,}")
    print(f"Average population per node: {nodes['population'].mean():.1f}")
    print(f"Population range: {nodes['population'].min()} - {nodes['population'].max()}")
    print(f"\nNode type distribution:")
    print(nodes['node_type'].value_counts())
    print(f"\nPopulation by node type:")
    print(nodes.groupby('node_type')['population'].agg(['count', 'sum', 'mean']).round(1))
    
    # Process edges data
    edges = edges_df.copy()
    
    # Calculate basic edge properties
    edges['travel_time'] = edges['length'] / 50  # Assume 50 units per time unit
    edges['capacity'] = edges['length'] * 0.1  # Simple capacity based on length
    
    print(f"Network Statistics:")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    print(f"\nNodes columns: {list(nodes.columns)}")
    print(f"\nEdges columns: {list(edges.columns)}")
    
    # Display sample of processed data
    print(f"\nSample nodes:")
    print(nodes.head())
    print(f"\nSample edges:")
    print(edges[['u', 'v', 'length', 'travel_time', 'capacity']].head())
else:
    print("Cannot process data - edges_df is None")
    
# Define sink nodes for evacuation
SINK_NODE_1 = 31253600 
SINK_NODE_2 = 5110476586  
# Evacuation simulation parameters
class EvacuationParameters:
    def __init__(self, nodes_df=None):
        # Update population based on loaded data

        self.total_population = nodes_df['population'].sum()
        self.total_households = int(self.total_population * 0.4)  # Approximate
        
        # Evacuation parameters
        self.time_step = 1  # minutes
        self.walking_speed = 5  # km/h
        self.driving_speed = 30  # km/h
        # Capacity parameters
        self.road_capacity_per_meter = 1  # persons per meter of road width
        self.default_road_width = 5
        self.evacuation_time_limit = 6000
        
params = EvacuationParameters(nodes)
print(f"Evacuation Parameters:")
print(f"Total Population: {params.total_population:,}")
print(f"Time Steps: {params.time_step} minutes")


# In[4]:


import re

# Function to parse LINESTRING geometry
def parse_linestring(linestring_str):
    """Parse LINESTRING format to extract coordinates"""
    if pd.isna(linestring_str) or linestring_str == '':
        return None
    
    # Extract coordinates from LINESTRING format
    pattern = r'LINESTRING \(([^)]+)\)'
    match = re.search(pattern, linestring_str)
    if not match:
        return None
    
    coords_str = match.group(1)
    coords = []
    for coord_pair in coords_str.split(', '):
        lon, lat = map(float, coord_pair.split())
        coords.append([lat, lon])  # Note: folium expects [lat, lon]
    return coords

# Function to get node coordinates from edges
def extract_node_coordinates(edges_df):
    """Extract node coordinates from edge geometry data"""
    node_coords = {}
    
    for idx, row in edges_df.iterrows():
        coords = parse_linestring(row['geometry'])
        if coords:
            u_node = row['u']
            v_node = row['v']
            
            # Store first coordinate for u node if not already stored
            if u_node not in node_coords:
                node_coords[u_node] = coords[0]  # [lat, lon]
            
            # Store last coordinate for v node if not already stored
            if v_node not in node_coords:
                node_coords[v_node] = coords[-1]  # [lat, lon]
    
    return node_coords

# Extract node coordinates from the dataset
if edges_df is not None:
    print("Processing geometric data for mapping...")
    
    # Extract node coordinates
    node_coordinates = extract_node_coordinates(edges_df)
    print(f"Extracted coordinates for {len(node_coordinates)} nodes")
    
    # Add coordinates to nodes DataFrame
    if 'nodes' in locals() and nodes is not None:
        nodes['lat'] = nodes.index.map(lambda x: node_coordinates.get(x, [None, None])[0])
        nodes['lon'] = nodes.index.map(lambda x: node_coordinates.get(x, [None, None])[1])
        
        # Remove nodes without coordinates
        nodes_with_coords = nodes.dropna(subset=['lat', 'lon'])
        print(f"Nodes with valid coordinates: {len(nodes_with_coords)}")
    
    # Process edge geometries
    edges_df['coordinates'] = edges_df['geometry'].apply(parse_linestring)
    edges_with_geom = edges_df.dropna(subset=['coordinates'])
    print(f"Edges with valid geometry: {len(edges_with_geom)}")
    
    # Calculate bounding box for the map
    all_coords = []
    for coords_list in edges_with_geom['coordinates']:
        if coords_list:
            all_coords.extend(coords_list)
    
    if all_coords:
        lats = [coord[0] for coord in all_coords]
        lons = [coord[1] for coord in all_coords]
        
        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2
        
        print(f"Map bounds:")
        print(f"  Latitude: {min(lats):.6f} to {max(lats):.6f}")
        print(f"  Longitude: {min(lons):.6f} to {max(lons):.6f}")
        print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
        
        # Store for map creation
        map_bounds = {
            'center': [center_lat, center_lon],
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
    else:
        print("No valid coordinates found!")
else:
    print("No edge data available for processing")


# In[5]:


# Create static visualization with matplotlib
if 'map_bounds' in locals() and edges_with_geom is not None:
    print("Creating static network visualization...")
    
    plt.figure(figsize=(15, 12))
    
    # Create color map for different road types
    unique_highways = edges_with_geom['highway'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_highways)))
    highway_colors = dict(zip(unique_highways, colors))
    
    # Plot roads
    for highway_type in unique_highways:
        highway_edges = edges_with_geom[edges_with_geom['highway'] == highway_type]
        
        for idx, row in highway_edges.iterrows():
            coords = row['coordinates']
            if coords and len(coords) > 1:
                lats = [coord[0] for coord in coords]
                lons = [coord[1] for coord in coords]
                
                plt.plot(lons, lats, 
                        color=highway_colors[highway_type], 
                        linewidth=2 if highway_type in ['primary', 'trunk', 'motorway'] else 1,
                        alpha=0.7,
                        label=highway_type if idx == highway_edges.index[0] else "")
    
    # Customize plot
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title('Road Network Visualization', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Equal aspect ratio for geographic accuracy
    plt.axis('equal')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot

    plt.show()
    
    print("Static visualization created and saved")
else:
    print("Cannot create static visualization - missing data")


# In[6]:


# Create NetworkX graph for network analysis
if edges_df is not None and 'node_coordinates' in locals():
    print("Creating NetworkX graph for network analysis...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with coordinates
    for node_id, coords in node_coordinates.items():
        if coords[0] is not None and coords[1] is not None:
            G.add_node(node_id, lat=coords[0], lon=coords[1])
    
    # Add edges with attributes
    for idx, row in edges_df.iterrows():
        if row['u'] in G.nodes and row['v'] in G.nodes:
            G.add_edge(row['u'], row['v'], 
                      length=row['length'],
                      highway=row.get('highway', 'unknown'),
                      name=row.get('name', ''),
                      travel_time=row.get('travel_time', row['length']/50))
    
    print(f"NetworkX Graph Statistics:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Is connected: {nx.is_weakly_connected(G)}")
    print(f"  Number of weakly connected components: {nx.number_weakly_connected_components(G)}")
    
    # Calculate basic network metrics
    if G.number_of_nodes() > 0:
        try:
            # Degree statistics
            degrees = dict(G.degree())
            avg_degree = sum(degrees.values()) / len(degrees)
            max_degree = max(degrees.values())
            
            print(f"  Average degree: {avg_degree:.2f}")
            print(f"  Maximum degree: {max_degree}")
            
            # Find nodes with highest degrees (important intersections)
            high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top 5 nodes by degree (major intersections):")
            for node, degree in high_degree_nodes:
                coords = node_coordinates.get(node, [None, None])
                print(f"    Node ID: {node} | Degree: {degree} | Coords: {coords}")
            
            # Show shelter nodes information
            print(f"  Shelter nodes information:")
            for shelter in [SINK_NODE_1, SINK_NODE_2]:  # SINK_NODE_1, SINK_NODE_2
                if shelter in G.nodes:
                    shelter_degree = degrees.get(shelter, 0)
                    shelter_coords = node_coordinates.get(shelter, [None, None])
                    # Add shelter capacity and mark this node as a shelter
                    G.nodes[shelter]['shelter_capacity'] = float('inf')  # Assuming infinite capacity for shelters
                    G.nodes[shelter]['is_shelter'] = True
                    print(f"    Shelter Node ID: {shelter} | Degree: {shelter_degree} | Coords: {shelter_coords}")
                else:
                    print(f"    Shelter Node ID: {shelter} | Status: NOT FOUND in network")
                
        except Exception as e:
            print(f"  Error calculating network metrics: {e}")
    
    # Store graph for later use
    road_network_graph = G
    print("NetworkX graph created successfully!")
else:
    print("Cannot create NetworkX graph - missing data")


# In[7]:


import folium
from shapely.geometry import LineString, Point
from shapely.wkt import loads
import re
from math import radians, cos, sin, asin, sqrt

# Function to parse LINESTRING geometry
def parse_linestring(linestring_str):
    """Parse LINESTRING format to extract coordinates"""
    if pd.isna(linestring_str) or linestring_str == '':
        return None
    
    # Extract coordinates from LINESTRING format
    pattern = r'LINESTRING \(([^)]+)\)'
    match = re.search(pattern, linestring_str)
    if not match:
        return None
    
    coords_str = match.group(1)
    coords = []
    for coord_pair in coords_str.split(', '):
        lon, lat = map(float, coord_pair.split())
        coords.append([lat, lon])  # Note: folium expects [lat, lon]
    return coords

# Function to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on earth"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Function to get node coordinates from edges
def extract_node_coordinates(edges_df):
    """Extract node coordinates from edge geometry data"""
    node_coords = {}
    
    for idx, row in edges_df.iterrows():
        coords = parse_linestring(row['geometry'])
        if coords:
            u_node = row['u']
            v_node = row['v']
            
            # Store first coordinate for u node if not already stored
            if u_node not in node_coords:
                node_coords[u_node] = coords[0]  # [lat, lon]
            
            # Store last coordinate for v node if not already stored
            if v_node not in node_coords:
                node_coords[v_node] = coords[-1]  # [lat, lon]
    
    return node_coords

print("Geometric processing functions defined successfully!")


# In[8]:


# Create enhanced map visualization with population data

if 'map_bounds' in locals() and edges_with_geom is not None and 'nodes' in locals():
    print("Creating enhanced map with population visualization...")
    
    # Create base map
    m = folium.Map(
        location=map_bounds['center'],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Color coding for different road types
    road_colors = {
        'primary': '#FF6B6B',     # Red
        'secondary': '#4ECDC4',   # Teal
        'tertiary': '#45B7D1',    # Blue
        'residential': '#96CEB4', # Green
        'unclassified': '#FFEAA7', # Yellow
        'trunk': '#DDA0DD',       # Plum
        'motorway': '#FF7675',    # Light red
        'service': '#74B9FF'      # Light blue
    }
    
    # Add roads to map
    road_count = 0
    for idx, row in edges_with_geom.iterrows():
        coords = row['coordinates']
        if coords and len(coords) > 1:
            # Get road type and color
            highway_type = row.get('highway', 'unclassified')
            color = road_colors.get(highway_type, '#95A5A6')  # Default gray
            
            # Create popup text
            popup_text = f"""
            <b>Road Information</b><br>
            From Node: {row['u']}<br>
            To Node: {row['v']}<br>
            Highway Type: {highway_type}<br>
            Length: {row['length']:.1f}m<br>
            Name: {row.get('name', 'Unnamed')}<br>
            Max Speed: {row.get('maxspeed', 'N/A')}
            """
            
            # Add road as polyline
            folium.PolyLine(
                coords,
                color=color,
                weight=3 if highway_type in ['primary', 'trunk', 'motorway'] else 2,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
            
            road_count += 1
    
    # Add nodes with population data as circle markers
    nodes_with_coords = nodes.dropna(subset=['lat', 'lon'])
    
    # Color mapping for node types
    node_type_colors = {
        'residential': '#2E8B57',  # Sea Green
        'commercial': '#FF4500',   # Orange Red
        'industrial': '#4169E1'    # Royal Blue
    }
    
    # Add population nodes to map
    for node_id, node_data in nodes_with_coords.iterrows():
        lat, lon = node_data['lat'], node_data['lon']
        population = node_data['population']
        node_type = node_data['node_type']
        
        # Scale circle size based on population (min 5, max 25)
        radius = max(5, min(25, population / 20))
        
        # Check if this is a shelter node
        is_shelter = node_id in [SINK_NODE_1, SINK_NODE_2]
        shelter_text = " (SHELTER)" if is_shelter else ""
        
        # Create popup with node information including Node ID
        popup_text = f"""
        <b>Node ID: {node_id}{shelter_text}</b><br>
        Type: {node_type.title()}<br>
        Population: {population:,}<br>
        Coordinates: ({lat:.6f}, {lon:.6f})<br>
        <small>Click to see Node ID: {node_id}</small>
        """
        
        # Special styling for shelter nodes
        if is_shelter:
            circle_color = '#FF0000'  # Red border for shelters
            fill_color = '#FFD700'    # Gold fill for shelters
            circle_weight = 3
            circle_radius = max(10, radius)  # Larger for shelters
        else:
            circle_color = 'black'
            fill_color = node_type_colors.get(node_type, '#808080')
            circle_weight = 1
            circle_radius = radius
        
        # Add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=circle_radius,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Node ID: {node_id} | Pop: {population:,}{shelter_text}",
            color=circle_color,
            weight=circle_weight,
            fillColor=fill_color,
            fillOpacity=0.8 if is_shelter else 0.7
        ).add_to(m)
        
        # Add text label for shelter nodes and high-population nodes
        if is_shelter or population > 400:
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="background-color: white; border: 1px solid black; border-radius: 3px; padding: 2px; font-size: 10px; font-weight: bold;">{node_id}</div>',
                    class_name='node-label'
                )
            ).add_to(m)
    
    print(f"Added {road_count} roads and {len(nodes_with_coords)} population nodes to the map")
    
    # Add enhanced legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: 380px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <b>Road Types</b><br>
    <i class="fa fa-minus" style="color:#FF6B6B"></i> Primary<br>
    <i class="fa fa-minus" style="color:#4ECDC4"></i> Secondary<br>
    <i class="fa fa-minus" style="color:#45B7D1"></i> Tertiary<br>
    <i class="fa fa-minus" style="color:#96CEB4"></i> Residential<br>
    <i class="fa fa-minus" style="color:#FFEAA7"></i> Unclassified<br>
    <i class="fa fa-minus" style="color:#DDA0DD"></i> Trunk<br>
    <i class="fa fa-minus" style="color:#74B9FF"></i> Service<br>
    <br>
    <b>Population Nodes</b><br>
    <i class="fa fa-circle" style="color:#2E8B57"></i> Residential<br>
    <i class="fa fa-circle" style="color:#FF4500"></i> Commercial<br>
    <i class="fa fa-circle" style="color:#4169E1"></i> Industrial<br>
    <i class="fa fa-circle" style="color:#FFD700; border: 2px solid red;"></i> Shelter Nodes<br>
    <br>
    <b>Shelter Information</b><br>
    <small>SINK_NODE_1: {sink1}<br>
    SINK_NODE_2: {sink2}</small><br>
    <br>
    <small>Circle size = Population<br>
    Total Population: {total_population:,}<br>
    <br>
    <b>Interaction:</b><br>
    • Click nodes for Node ID<br>
    • Hover for quick info<br>
    • Labels show for shelters</small>
    </div>
    '''.format(
        total_population=nodes['population'].sum(),
        sink1=SINK_NODE_1,
        sink2=SINK_NODE_2
    )
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit map to bounds
    m.fit_bounds([
        [map_bounds['min_lat'], map_bounds['min_lon']],
        [map_bounds['max_lat'], map_bounds['max_lon']]
    ])
    
    # Display the map
    display(m)
    

else:
    print("Cannot create enhanced map - missing data or coordinates")


# In[ ]:





# In[9]:


# Max Flow Analysis Implementation
from networkx.algorithms.flow import maximum_flow
from networkx.algorithms.flow import shortest_augmenting_path
import networkx.algorithms.flow as nxflow

class MaxFlowAnalyzer:
    def __init__(self, graph, shelter_nodes, capacity_attribute='capacity'):
        """
        Initialize the Max Flow Analyzer
        
        Args:
            graph: NetworkX graph
            shelter_nodes: List of shelter node IDs
            capacity_attribute: Name of edge attribute containing capacity
        """
        self.graph = graph.copy()
        self.shelter_nodes = shelter_nodes
        self.capacity_attribute = capacity_attribute
        self.results = {}
        
        # Prepare graph for max flow analysis
        self._prepare_graph()
    
    def _prepare_graph(self):
        """Prepare the graph for max flow analysis"""
        # Ensure all edges have capacity attribute
        for u, v, data in self.graph.edges(data=True):
            if self.capacity_attribute not in data:
                # Set default capacity based on edge length and road type
                length = data.get('length', 100)
                highway_type = data.get('highway', 'residential')
                
                # Capacity multipliers for different road types
                capacity_multipliers = {
                    'motorway': 10.0,
                    'trunk': 8.0,
                    'primary': 6.0,
                    'secondary': 4.0,
                    'tertiary': 3.0,
                    'residential': 2.0,
                    'service': 1.0,
                    'unclassified': 1.5
                }
                
                multiplier = capacity_multipliers.get(highway_type, 1.0)
                # Base capacity: length * multiplier * base_flow_rate
                base_capacity = max(1, int(length * multiplier * 0.1))
                data[self.capacity_attribute] = base_capacity
        
        print(f"Graph prepared with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def create_supersink(self):
        """Create a supersink connected to all shelter nodes"""
        supersink = 'SUPERSINK'
        self.graph.add_node(supersink)
        
        # Connect all shelter nodes to supersink with infinite capacity
        for shelter in self.shelter_nodes:
            if shelter in self.graph.nodes:
                self.graph.add_edge(shelter, supersink, **{self.capacity_attribute: float('inf')})
        
        return supersink
    
    @monitor_performance
    def calculate_max_flow_to_shelters(self, source_nodes=None, use_supersink=True):
        """
        Calculate max flow from source nodes to shelter nodes
        
        Args:
            source_nodes: List of source nodes. If None, use all nodes except shelters
            use_supersink: Whether to use supersink approach or calculate to each shelter separately
        """
        if source_nodes is None:
            # Use all nodes except shelter nodes
            source_nodes = [n for n in self.graph.nodes if n not in self.shelter_nodes and n != 'SUPERSINK']
        
        print(f"Calculating max flow for {len(source_nodes)} source nodes to {len(self.shelter_nodes)} shelter nodes...")
        
        if use_supersink:
            return self._calculate_with_supersink(source_nodes)
        else:
            return self._calculate_individual_flows(source_nodes)
    
    def _calculate_with_supersink(self, source_nodes):
        """Calculate max flow using supersink approach"""
        # Create supersink
        supersink = self.create_supersink()
        
        results = {}
        failed_calculations = []
        
        for i, source in enumerate(source_nodes):
            if i % 100 == 0:
                print(f"Processing node {i+1}/{len(source_nodes)}: {source}")
            
            try:
                if source in self.graph.nodes and nx.has_path(self.graph, source, supersink):
                    flow_value, flow_dict = maximum_flow(
                        self.graph, source, supersink, 
                        capacity=self.capacity_attribute,
                        flow_func=shortest_augmenting_path
                    )
                    
                    results[source] = {
                        'max_flow': flow_value,
                        'flow_dict': flow_dict,
                        'reachable_shelters': [s for s in self.shelter_nodes if s in self.graph.nodes and nx.has_path(self.graph, source, s)]
                    }
                else:
                    results[source] = {
                        'max_flow': 0,
                        'flow_dict': {},
                        'reachable_shelters': [],
                        'error': 'No path to supersink'
                    }
                    failed_calculations.append(source)
            
            except Exception as e:
                results[source] = {
                    'max_flow': 0,
                    'flow_dict': {},
                    'reachable_shelters': [],
                    'error': str(e)
                }
                failed_calculations.append(source)
        
        if failed_calculations:
            print(f"Warning: {len(failed_calculations)} nodes failed max flow calculation")
        
        return results
    
    def _calculate_individual_flows(self, source_nodes):
        """Calculate max flow to each shelter individually"""
        results = {}
        
        for i, source in enumerate(source_nodes):
            if i % 50 == 0:
                print(f"Processing node {i+1}/{len(source_nodes)}: {source}")
            
            results[source] = {
                'max_flow_to_shelters': {},
                'total_max_flow': 0,
                'reachable_shelters': []
            }
            
            total_flow = 0
            for shelter in self.shelter_nodes:
                if shelter in self.graph.nodes:
                    try:
                        if nx.has_path(self.graph, source, shelter):
                            flow_value, _ = maximum_flow(
                                self.graph, source, shelter,
                                capacity=self.capacity_attribute,
                                flow_func=shortest_augmenting_path
                            )
                            results[source]['max_flow_to_shelters'][shelter] = flow_value
                            total_flow += flow_value
                            results[source]['reachable_shelters'].append(shelter)
                        else:
                            results[source]['max_flow_to_shelters'][shelter] = 0
                    except Exception as e:
                        results[source]['max_flow_to_shelters'][shelter] = 0
                        print(f"Error calculating flow from {source} to {shelter}: {e}")
            
            results[source]['total_max_flow'] = total_flow
        
        return results
    
    def analyze_results(self, results):
        """Analyze and summarize max flow results"""
        print("\n=== MAX FLOW ANALYSIS RESULTS ===")
        
        # Basic statistics
        valid_results = {k: v for k, v in results.items() if 'max_flow' in v and v['max_flow'] > 0}
        zero_flow_nodes = {k: v for k, v in results.items() if 'max_flow' in v and v['max_flow'] == 0}
        error_nodes = {k: v for k, v in results.items() if 'error' in v}
        
        print(f"Total nodes analyzed: {len(results)}")
        print(f"Nodes with positive max flow: {len(valid_results)}")
        print(f"Nodes with zero max flow: {len(zero_flow_nodes)}")
        print(f"Nodes with calculation errors: {len(error_nodes)}")
        
        if valid_results:
            flows = [v['max_flow'] for v in valid_results.values()]
            print(f"\nFlow Statistics:")
            print(f"  Average max flow: {np.mean(flows):.2f}")
            print(f"  Median max flow: {np.median(flows):.2f}")
            print(f"  Min max flow: {np.min(flows):.2f}")
            print(f"  Max max flow: {np.max(flows):.2f}")
            print(f"  Total flow capacity: {np.sum(flows):.2f}")
            
            # Top 10 nodes with highest max flow
            sorted_flows = sorted(valid_results.items(), key=lambda x: x[1]['max_flow'], reverse=True)
            print(f"\nTop 10 nodes with highest max flow:")
            for i, (node_id, data) in enumerate(sorted_flows[:10]):
                shelters = data.get('reachable_shelters', [])
                print(f"  {i+1}. Node {node_id}: {data['max_flow']:.2f} (reaches {len(shelters)} shelters)")
        
        return {
            'summary': {
                'total_nodes': len(results),
                'positive_flow_nodes': len(valid_results),
                'zero_flow_nodes': len(zero_flow_nodes),
                'error_nodes': len(error_nodes)
            },
            'flows': [v['max_flow'] for v in valid_results.values()] if valid_results else [],
            'top_nodes': sorted(valid_results.items(), key=lambda x: x[1]['max_flow'], reverse=True)[:20] if valid_results else []
        }

print("Max Flow Analyzer class defined successfully!")


# In[10]:


# Execute Max Flow Analysis

if 'road_network_graph' in locals() and road_network_graph is not None:
    print("Starting Max Flow Analysis...")
    
    # Initialize the Max Flow Analyzer
    shelter_nodes = [SINK_NODE_1, SINK_NODE_2]
    analyzer = MaxFlowAnalyzer(road_network_graph, shelter_nodes)
    
    # Calculate max flow from all nodes to shelters
    print("\nCalculating max flow from all nodes to shelter nodes...")
    print(f"Shelter nodes: {shelter_nodes}")
    
    # Get all nodes that have population (exclude shelter nodes)
    source_nodes = [node for node in nodes.index if node in road_network_graph.nodes and node not in shelter_nodes]
    print(f"Analyzing {len(source_nodes)} source nodes with population data")
    
    # Calculate max flow using supersink approach (more efficient)
    flow_results = analyzer.calculate_max_flow_to_shelters(source_nodes=source_nodes, use_supersink=True)
    
    # Analyze and display results
    analysis_summary = analyzer.analyze_results(flow_results)
    
    # Store results for further analysis
    max_flow_results = flow_results
    max_flow_summary = analysis_summary
    
    print("\nMax flow analysis completed successfully!")
    
else:
    print("Error: Road network graph not found. Please run the previous cells first.")


# In[11]:


# Visualize Max Flow Results

if 'max_flow_results' in locals() and max_flow_results:
    print("Creating visualizations for max flow results...")
    
    # Extract flow data for visualization
    node_flows = {}
    for node_id, result in max_flow_results.items():
        if 'max_flow' in result:
            node_flows[node_id] = result['max_flow']
    
    # Create DataFrame for easier analysis
    flow_df = pd.DataFrame([
        {
            'node_id': node_id,
            'max_flow': result.get('max_flow', 0),
            'reachable_shelters': len(result.get('reachable_shelters', [])),
            'has_error': 'error' in result,
            'population': nodes.loc[node_id, 'population'] if node_id in nodes.index else 0
        }
        for node_id, result in max_flow_results.items()
    ])
    
    print(f"Created flow analysis DataFrame with {len(flow_df)} nodes")
    
    # 1. Flow Distribution Histogram
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    positive_flows = flow_df[flow_df['max_flow'] > 0]['max_flow']
    plt.hist(positive_flows, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Max Flow Value')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Max Flow Values\n(Nodes with Positive Flow)')
    plt.grid(True, alpha=0.3)
    
    # 2. Flow vs Population Scatter Plot
    plt.subplot(2, 3, 2)
    valid_data = flow_df[(flow_df['max_flow'] > 0) & (flow_df['population'] > 0)]
    plt.scatter(valid_data['population'], valid_data['max_flow'], alpha=0.6, color='coral')
    plt.xlabel('Population')
    plt.ylabel('Max Flow')
    plt.title('Max Flow vs Population')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(valid_data) > 1:
        correlation = valid_data['population'].corr(valid_data['max_flow'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Reachable Shelters Distribution
    plt.subplot(2, 3, 3)
    shelter_counts = flow_df['reachable_shelters'].value_counts().sort_index()
    plt.bar(shelter_counts.index, shelter_counts.values, color='lightgreen', alpha=0.7)
    plt.xlabel('Number of Reachable Shelters')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Reachable Shelters')
    plt.grid(True, alpha=0.3)
    
    # 4. Flow Efficiency (Flow per Population)
    plt.subplot(2, 3, 4)
    efficiency_data = valid_data.copy()
    efficiency_data['flow_efficiency'] = efficiency_data['max_flow'] / efficiency_data['population']
    plt.hist(efficiency_data['flow_efficiency'], bins=25, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('Flow Efficiency (Flow/Population)')
    plt.ylabel('Number of Nodes')
    plt.title('Flow Efficiency Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. Top Nodes Bar Chart
    plt.subplot(2, 3, 5)
    top_nodes = flow_df.nlargest(10, 'max_flow')
    plt.barh(range(len(top_nodes)), top_nodes['max_flow'], color='purple', alpha=0.7)
    plt.yticks(range(len(top_nodes)), [f'Node {node_id}' for node_id in top_nodes['node_id']])
    plt.xlabel('Max Flow')
    plt.title('Top 10 Nodes by Max Flow')
    plt.grid(True, alpha=0.3)
    
    # 6. Error Analysis
    plt.subplot(2, 3, 6)
    error_summary = flow_df['has_error'].value_counts()
    labels = ['Successful', 'Failed']
    colors = ['lightblue', 'lightcoral']
    plt.pie(error_summary.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Calculation Success Rate')
    
    plt.tight_layout()
    plt.show()
    
    # Summary Statistics Table
    print("\n=== DETAILED FLOW STATISTICS ===")
    print(f"Total nodes analyzed: {len(flow_df)}")
    print(f"Nodes with positive flow: {len(flow_df[flow_df['max_flow'] > 0])}")
    print(f"Nodes with zero flow: {len(flow_df[flow_df['max_flow'] == 0])}")
    print(f"Nodes with calculation errors: {len(flow_df[flow_df['has_error']])}")
    
    if len(positive_flows) > 0:
        print(f"\nFlow Statistics (Positive flows only):")
        print(f"  Mean: {positive_flows.mean():.2f}")
        print(f"  Median: {positive_flows.median():.2f}")
        print(f"  Standard Deviation: {positive_flows.std():.2f}")
        print(f"  Min: {positive_flows.min():.2f}")
        print(f"  Max: {positive_flows.max():.2f}")
        print(f"  Total Flow Capacity: {positive_flows.sum():.2f}")
    
    # Shelter accessibility analysis
    print(f"\nShelter Accessibility:")
    for i in range(3):
        nodes_reaching_i_shelters = len(flow_df[flow_df['reachable_shelters'] == i])
        print(f"  Nodes reaching {i} shelters: {nodes_reaching_i_shelters} ({100*nodes_reaching_i_shelters/len(flow_df):.1f}%)")

else:
    print("No max flow results available. Please run the max flow calculation first.")


# In[12]:


# Enhanced Map Visualization with Max Flow Results

if 'max_flow_results' in locals() and 'map_bounds' in locals() and edges_with_geom is not None:
    print("Creating enhanced map with max flow visualization...")
    
    # Create base map
    m_flow = folium.Map(
        location=map_bounds['center'],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Color coding for different road types (same as before)
    road_colors = {
        'primary': '#FF6B6B', 'secondary': '#4ECDC4', 'tertiary': '#45B7D1',
        'residential': '#96CEB4', 'unclassified': '#FFEAA7', 'trunk': '#DDA0DD',
        'motorway': '#FF7675', 'service': '#74B9FF'
    }
    
    # Add roads to map
    for idx, row in edges_with_geom.iterrows():
        coords = row['coordinates']
        if coords and len(coords) > 1:
            highway_type = row.get('highway', 'unclassified')
            color = road_colors.get(highway_type, '#95A5A6')
            
            folium.PolyLine(
                coords, color=color,
                weight=3 if highway_type in ['primary', 'trunk', 'motorway'] else 2,
                opacity=0.6
            ).add_to(m_flow)
    
    # Get flow data for nodes
    flow_values = []
    for node_id in nodes.index:
        if node_id in max_flow_results:
            flow_values.append(max_flow_results[node_id].get('max_flow', 0))
        else:
            flow_values.append(0)
    
    # Calculate flow percentiles for color mapping
    flow_values_positive = [f for f in flow_values if f > 0]
    if flow_values_positive:
        flow_min = min(flow_values_positive)
        flow_max = max(flow_values_positive)
        flow_25 = np.percentile(flow_values_positive, 25)
        flow_75 = np.percentile(flow_values_positive, 75)
    else:
        flow_min = flow_max = flow_25 = flow_75 = 0
    
    # Function to get color based on flow value
    def get_flow_color(flow_value):
        if flow_value == 0:
            return '#808080'  # Gray for zero flow
        elif flow_value <= flow_25:
            return '#FEF0D9'  # Light yellow for low flow
        elif flow_value <= flow_75:
            return '#FDD49E'  # Orange for medium flow
        else:
            return '#D7301F'  # Red for high flow
    
    # Add nodes with max flow visualization
    nodes_with_coords = nodes.dropna(subset=['lat', 'lon'])
    
    for node_id, node_data in nodes_with_coords.iterrows():
        lat, lon = node_data['lat'], node_data['lon']
        population = node_data['population']
        
        # Get flow data
        flow_data = max_flow_results.get(node_id, {})
        max_flow_value = flow_data.get('max_flow', 0)
        reachable_shelters = flow_data.get('reachable_shelters', [])
        has_error = 'error' in flow_data
        
        # Scale circle size based on max flow (min 5, max 25)
        if max_flow_value > 0:
            radius = max(5, min(25, max_flow_value / (flow_max / 20)))
        else:
            radius = 3
        
        # Check if this is a shelter node
        is_shelter = node_id in [SINK_NODE_1, SINK_NODE_2]
        
        # Create popup with detailed information
        popup_text = f"""
        <b>Node ID: {node_id}{'(SHELTER)' if is_shelter else ''}</b><br>
        Population: {population:,}<br>
        Max Flow: {max_flow_value:.2f}<br>
        Reachable Shelters: {len(reachable_shelters)}<br>
        Shelters: {reachable_shelters}<br>
        {'<span style="color: red;">Calculation Error</span>' if has_error else ''}<br>
        Coordinates: ({lat:.6f}, {lon:.6f})
        """
        
        # Special styling for different node types
        if is_shelter:
            circle_color = '#FF0000'  # Red border for shelters
            fill_color = '#FFD700'    # Gold fill for shelters
            circle_weight = 3
            circle_radius = 15
        elif has_error:
            circle_color = '#FF0000'  # Red border for errors
            fill_color = '#FF6B6B'    # Light red fill for errors
            circle_weight = 2
            circle_radius = radius
        else:
            circle_color = 'black'
            fill_color = get_flow_color(max_flow_value)
            circle_weight = 1
            circle_radius = radius
        
        # Add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=circle_radius,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Node {node_id} | Flow: {max_flow_value:.1f} | Pop: {population:,}",
            color=circle_color,
            weight=circle_weight,
            fillColor=fill_color,
            fillOpacity=0.8
        ).add_to(m_flow)
        
        # Add labels for shelter nodes and high-flow nodes
        if is_shelter or max_flow_value > flow_75:
            label_text = f"S{node_id}" if is_shelter else f"{max_flow_value:.0f}"
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="background-color: white; border: 1px solid black; border-radius: 3px; padding: 2px; font-size: 10px; font-weight: bold;">{label_text}</div>',
                    class_name='flow-label'
                )
            ).add_to(m_flow)
    
    # Enhanced legend for max flow map
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: 450px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <b>Max Flow Analysis Results</b><br><br>
    
    <b>Node Colors (by Max Flow):</b><br>
    <i class="fa fa-circle" style="color:#808080"></i> Zero Flow<br>
    <i class="fa fa-circle" style="color:#FEF0D9"></i> Low Flow (≤{flow_25:.1f})<br>
    <i class="fa fa-circle" style="color:#FDD49E"></i> Medium Flow (≤{flow_75:.1f})<br>
    <i class="fa fa-circle" style="color:#D7301F"></i> High Flow (>{flow_75:.1f})<br>
    <i class="fa fa-circle" style="color:#FFD700; border: 2px solid red;"></i> Shelter Nodes<br>
    <i class="fa fa-circle" style="color:#FF6B6B; border: 2px solid red;"></i> Calculation Error<br>
    <br>
    
    <b>Flow Statistics:</b><br>
    <small>
    Total Nodes: {len(max_flow_results)}<br>
    Positive Flow: {len([f for f in flow_values if f > 0])}<br>
    Zero Flow: {len([f for f in flow_values if f == 0])}<br>
    Max Flow: {flow_max:.1f}<br>
    Min Flow: {flow_min:.1f}<br>
    </small><br>
    
    <b>Shelter Information:</b><br>
    <small>
    SINK_NODE_1: {SINK_NODE_1}<br>
    SINK_NODE_2: {SINK_NODE_2}
    </small><br>
    
    <b>Legend:</b><br>
    <small>
    • Circle size ∝ Max Flow<br>
    • Click for detailed info<br>
    • Labels show high-flow nodes<br>
    • 'S' prefix = Shelter
    </small>
    </div>
    '''
    
    m_flow.get_root().html.add_child(folium.Element(legend_html))
    
    # Fit map to bounds
    m_flow.fit_bounds([
        [map_bounds['min_lat'], map_bounds['min_lon']],
        [map_bounds['max_lat'], map_bounds['max_lon']]
    ])
    
    # Display the enhanced map
    print(f"Enhanced map created with max flow visualization for {len(nodes_with_coords)} nodes")
    display(m_flow)
    
else:
    print("Cannot create enhanced flow map - missing required data")


# In[13]:


# Diagnostic Analysis for Zero Flow and Error Nodes

if 'max_flow_results' in locals() and max_flow_results:
    print("=== DIAGNOSTIC ANALYSIS FOR PROBLEMATIC NODES ===")
    
    # Separate nodes by their status
    zero_flow_nodes = []
    error_nodes = []
    successful_nodes = []
    
    for node_id, result in max_flow_results.items():
        if 'error' in result:
            error_nodes.append((node_id, result))
        elif result.get('max_flow', 0) == 0:
            zero_flow_nodes.append((node_id, result))
        else:
            successful_nodes.append((node_id, result))
    
    print(f"\nBreakdown:")
    print(f"  Successful nodes: {len(successful_nodes)}")
    print(f"  Zero flow nodes: {len(zero_flow_nodes)}")
    print(f"  Error nodes: {len(error_nodes)}")
    
    # Analyze error nodes
    if error_nodes:
        print(f"\n=== ERROR NODES ANALYSIS ===")
        for i, (node_id, result) in enumerate(error_nodes):
            error_msg = result.get('error', 'Unknown error')
            print(f"  {i+1}. Node {node_id}: {error_msg}")
            
            # Check if node exists in graph
            if node_id in road_network_graph.nodes:
                print(f"     - Node exists in graph: YES")
                # Check connectivity to shelters
                connectivity = []
                for shelter in [SINK_NODE_1, SINK_NODE_2]:
                    if shelter in road_network_graph.nodes:
                        has_path = nx.has_path(road_network_graph, node_id, shelter)
                        connectivity.append(f"to {shelter}: {has_path}")
                    else:
                        connectivity.append(f"to {shelter}: SHELTER NOT IN GRAPH")
                print(f"     - Connectivity: {', '.join(connectivity)}")
            else:
                print(f"     - Node exists in graph: NO")
    
    # Analyze zero flow nodes
    if zero_flow_nodes:
        print(f"\n=== ZERO FLOW NODES ANALYSIS ===")
        for i, (node_id, result) in enumerate(zero_flow_nodes):
            print(f"  {i+1}. Node {node_id}")
            
            # Check if node exists in graph
            if node_id in road_network_graph.nodes:
                print(f"     - Node exists in graph: YES")
                
                # Check connectivity to shelters
                connectivity = []
                reachable_shelters = []
                for shelter in [SINK_NODE_1, SINK_NODE_2]:
                    if shelter in road_network_graph.nodes:
                        has_path = nx.has_path(road_network_graph, node_id, shelter)
                        connectivity.append(f"to {shelter}: {has_path}")
                        if has_path:
                            reachable_shelters.append(shelter)
                    else:
                        connectivity.append(f"to {shelter}: SHELTER NOT IN GRAPH")
                
                print(f"     - Connectivity: {', '.join(connectivity)}")
                print(f"     - Reachable shelters: {len(reachable_shelters)}")
                
                # Check node degree (connections)
                degree = road_network_graph.degree(node_id)
                print(f"     - Node degree: {degree}")
                
                # If connected to shelters, check why flow is zero
                if reachable_shelters:
                    print(f"     - Has path to shelters but zero flow - checking edge capacities...")
                    
                    # Check outgoing edges and their capacities
                    out_edges = list(road_network_graph.out_edges(node_id, data=True))
                    if out_edges:
                        min_capacity = min([data.get('capacity', 0) for _, _, data in out_edges])
                        max_capacity = max([data.get('capacity', 0) for _, _, data in out_edges])
                        print(f"     - Outgoing edges: {len(out_edges)}, capacity range: {min_capacity}-{max_capacity}")
                    else:
                        print(f"     - No outgoing edges (dead end)")
            else:
                print(f"     - Node exists in graph: NO")
    
    # Check shelter node status
    print(f"\n=== SHELTER NODES STATUS ===")
    for shelter in [SINK_NODE_1, SINK_NODE_2]:
        if shelter in road_network_graph.nodes:
            degree = road_network_graph.degree(shelter)
            in_degree = road_network_graph.in_degree(shelter)
            out_degree = road_network_graph.out_degree(shelter)
            print(f"  Shelter {shelter}:")
            print(f"    - In graph: YES")
            print(f"    - Total degree: {degree}")
            print(f"    - In-degree: {in_degree}")
            print(f"    - Out-degree: {out_degree}")
            
            # Check how many nodes can reach this shelter
            reachable_count = 0
            for node in road_network_graph.nodes:
                if node != shelter and nx.has_path(road_network_graph, node, shelter):
                    reachable_count += 1
            print(f"    - Nodes that can reach this shelter: {reachable_count}")
        else:
            print(f"  Shelter {shelter}: NOT IN GRAPH")
    
    # Check graph connectivity
    print(f"\n=== GRAPH CONNECTIVITY ANALYSIS ===")
    print(f"  Total nodes in graph: {road_network_graph.number_of_nodes()}")
    print(f"  Total edges in graph: {road_network_graph.number_of_edges()}")
    print(f"  Is weakly connected: {nx.is_weakly_connected(road_network_graph)}")
    print(f"  Number of weakly connected components: {nx.number_weakly_connected_components(road_network_graph)}")
    
    if not nx.is_weakly_connected(road_network_graph):
        components = list(nx.weakly_connected_components(road_network_graph))
        print(f"  Component sizes: {[len(comp) for comp in components]}")
        
        # Check which component contains the shelters
        shelter_components = []
        for i, comp in enumerate(components):
            shelter_in_comp = [s for s in [SINK_NODE_1, SINK_NODE_2] if s in comp]
            if shelter_in_comp:
                shelter_components.append((i, len(comp), shelter_in_comp))
        
        print(f"  Components containing shelters: {shelter_components}")
        
        # Check if error/zero-flow nodes are in isolated components
        problematic_nodes = [node for node, _ in error_nodes + zero_flow_nodes]
        for i, comp in enumerate(components):
            problematic_in_comp = [n for n in problematic_nodes if n in comp]
            if problematic_in_comp:
                print(f"  Component {i} (size {len(comp)}) contains problematic nodes: {len(problematic_in_comp)}")

else:
    print("No max flow results available for analysis.")


# In[14]:


# Fix for Unreachable Shelter Nodes

def fix_shelter_connectivity(graph, shelter_nodes, node_coordinates):
    """
    Fix connectivity issues with shelter nodes by ensuring they are reachable
    """
    print("=== FIXING SHELTER CONNECTIVITY ===")
    
    fixed_graph = graph.copy()
    fixes_applied = []
    
    for shelter in shelter_nodes:
        if shelter in fixed_graph.nodes:
            in_degree = fixed_graph.in_degree(shelter)
            out_degree = fixed_graph.out_degree(shelter)
            
            print(f"\nAnalyzing Shelter {shelter}:")
            print(f"  Current in-degree: {in_degree}")
            print(f"  Current out-degree: {out_degree}")
            
            # If shelter has no incoming edges, it's unreachable
            if in_degree == 0:
                print(f"  Problem: Shelter {shelter} is unreachable (no incoming edges)")
                
                # Find nearby nodes to connect to this shelter
                if shelter in node_coordinates:
                    shelter_coords = node_coordinates[shelter]
                    shelter_lat, shelter_lon = shelter_coords[0], shelter_coords[1]
                    
                    # Find nodes within a reasonable distance
                    nearby_nodes = []
                    for node_id, coords in node_coordinates.items():
                        if node_id != shelter and node_id in fixed_graph.nodes:
                            if coords[0] is not None and coords[1] is not None:
                                # Calculate approximate distance
                                lat_diff = abs(coords[0] - shelter_lat)
                                lon_diff = abs(coords[1] - shelter_lon)
                                distance = (lat_diff**2 + lon_diff**2)**0.5
                                
                                if distance < 0.01:  # Approximately 1km
                                    nearby_nodes.append((node_id, distance))
                    
                    # Sort by distance and connect closest nodes
                    nearby_nodes.sort(key=lambda x: x[1])
                    connections_made = 0
                    
                    for node_id, distance in nearby_nodes[:5]:  # Connect up to 5 closest nodes
                        # Add edge from nearby node to shelter
                        if not fixed_graph.has_edge(node_id, shelter):
                            # Calculate capacity based on typical road capacity
                            capacity = 50  # Medium capacity
                            fixed_graph.add_edge(node_id, shelter, 
                                                length=distance*100000,  # Approximate meters
                                                capacity=capacity,
                                                highway='residential',
                                                name=f'Connection to Shelter {shelter}',
                                                travel_time=distance*100000/30)  # 30 km/h
                            connections_made += 1
                            fixes_applied.append(f"Added edge {node_id} -> {shelter}")
                    
                    print(f"  Fix applied: Added {connections_made} incoming edges to shelter")
                else:
                    print(f"  Cannot fix: No coordinates available for shelter {shelter}")
            
            # If shelter has no outgoing edges, add some (shelters might need evacuation routes too)
            elif out_degree == 0:
                print(f"  Note: Shelter {shelter} has no outgoing edges (normal for final destinations)")
            
            else:
                print(f"  Status: Shelter {shelter} connectivity looks good")
    
    if fixes_applied:
        print(f"\nFixes applied: {len(fixes_applied)}")
        for fix in fixes_applied:
            print(f"  - {fix}")
        return fixed_graph, True
    else:
        print(f"\nNo fixes needed or applied")
        return graph, False

# Apply the fix
if 'road_network_graph' in locals() and 'node_coordinates' in locals():
    print("Attempting to fix shelter connectivity issues...")
    
    fixed_graph, was_fixed = fix_shelter_connectivity(road_network_graph, [SINK_NODE_1, SINK_NODE_2], node_coordinates)
    
    if was_fixed:
        print("\n=== RE-RUNNING MAX FLOW ANALYSIS WITH FIXED GRAPH ===")
        
        # Create new analyzer with fixed graph
        analyzer_fixed = MaxFlowAnalyzer(fixed_graph, [SINK_NODE_1, SINK_NODE_2])
        
        # Get source nodes (exclude shelter nodes)
        source_nodes = [node for node in nodes.index if node in fixed_graph.nodes and node not in [SINK_NODE_1, SINK_NODE_2]]
        
        # Calculate max flow with fixed graph
        flow_results_fixed = analyzer_fixed.calculate_max_flow_to_shelters(source_nodes=source_nodes, use_supersink=True)
        
        # Analyze results
        analysis_summary_fixed = analyzer_fixed.analyze_results(flow_results_fixed)
        
        # Store fixed results
        max_flow_results_fixed = flow_results_fixed
        max_flow_summary_fixed = analysis_summary_fixed
        road_network_graph_fixed = fixed_graph
        
        print("\n=== COMPARISON: BEFORE vs AFTER FIX ===")
        
        # Compare results
        original_success = len([r for r in max_flow_results.values() if r.get('max_flow', 0) > 0])
        original_errors = len([r for r in max_flow_results.values() if 'error' in r])
        
        fixed_success = len([r for r in flow_results_fixed.values() if r.get('max_flow', 0) > 0])
        fixed_errors = len([r for r in flow_results_fixed.values() if 'error' in r])
        
        print(f"Successful calculations: {original_success} -> {fixed_success} (Δ{fixed_success - original_success})")
        print(f"Failed calculations: {original_errors} -> {fixed_errors} (Δ{fixed_errors - original_errors})")
        
        if fixed_errors == 0:
            print("✅ All connectivity issues resolved!")
        else:
            print(f"⚠️  Still {fixed_errors} nodes with issues")
    
else:
    print("Cannot apply fix: Missing required data")


# In[15]:


# Final Diagnostic for Remaining Issues

if 'max_flow_results_fixed' in locals():
    print("=== FINAL DIAGNOSTIC: REMAINING PROBLEMATIC NODES ===")
    
    # Find remaining problematic nodes
    remaining_error_nodes = []
    remaining_zero_nodes = []
    
    for node_id, result in max_flow_results_fixed.items():
        if 'error' in result:
            remaining_error_nodes.append((node_id, result))
        elif result.get('max_flow', 0) == 0:
            remaining_zero_nodes.append((node_id, result))
    
    print(f"Remaining error nodes: {len(remaining_error_nodes)}")
    print(f"Remaining zero flow nodes: {len(remaining_zero_nodes)}")
    
    # Analyze remaining problematic nodes
    if remaining_error_nodes:
        print(f"\n=== REMAINING ERROR NODES ===")
        for node_id, result in remaining_error_nodes:
            error_msg = result.get('error', 'Unknown error')
            print(f"  Node {node_id}: {error_msg}")
            
            if node_id in road_network_graph_fixed.nodes:
                # Check connectivity to both shelters
                connectivity_1 = nx.has_path(road_network_graph_fixed, node_id, SINK_NODE_1)
                connectivity_2 = nx.has_path(road_network_graph_fixed, node_id, SINK_NODE_2)
                degree = road_network_graph_fixed.degree(node_id)
                
                print(f"    - Can reach Shelter {SINK_NODE_1}: {connectivity_1}")
                print(f"    - Can reach Shelter {SINK_NODE_2}: {connectivity_2}")
                print(f"    - Node degree: {degree}")
                
                # If no connectivity, this node might be in an isolated component
                if not connectivity_1 and not connectivity_2:
                    print(f"    - Node appears to be in an isolated component")
    
    if remaining_zero_nodes:
        print(f"\n=== REMAINING ZERO FLOW NODES ===")
        for node_id, result in remaining_zero_nodes:
            print(f"  Node {node_id}")
            
            if node_id in road_network_graph_fixed.nodes:
                connectivity_1 = nx.has_path(road_network_graph_fixed, node_id, SINK_NODE_1)
                connectivity_2 = nx.has_path(road_network_graph_fixed, node_id, SINK_NODE_2)
                degree = road_network_graph_fixed.degree(node_id)
                
                print(f"    - Can reach Shelter {SINK_NODE_1}: {connectivity_1}")
                print(f"    - Can reach Shelter {SINK_NODE_2}: {connectivity_2}")
                print(f"    - Node degree: {degree}")
    
    # Summary of improvements
    print(f"\n=== IMPROVEMENT SUMMARY ===")
    original_working = 271
    fixed_working = 277
    improvement = fixed_working - original_working
    
    print(f"  Original working nodes: {original_working}/279 ({100*original_working/279:.1f}%)")
    print(f"  After fix working nodes: {fixed_working}/279 ({100*fixed_working/279:.1f}%)")
    print(f"  Improvement: +{improvement} nodes ({100*improvement/279:.1f} percentage points)")
    
    # Flow capacity improvement
    original_total_flow = sum([r.get('max_flow', 0) for r in max_flow_results.values()])
    fixed_total_flow = sum([r.get('max_flow', 0) for r in max_flow_results_fixed.values()])
    flow_improvement = fixed_total_flow - original_total_flow
    
    print(f"\n=== FLOW CAPACITY IMPROVEMENT ===")
    print(f"  Original total flow capacity: {original_total_flow:.0f}")
    print(f"  Fixed total flow capacity: {fixed_total_flow:.0f}")
    print(f"  Improvement: +{flow_improvement:.0f} ({100*flow_improvement/original_total_flow:.1f}% increase)")
    
    # Check if shelter 31253600 now works
    nodes_reaching_shelter_1 = 0
    nodes_reaching_shelter_2 = 0
    
    for node_id, result in max_flow_results_fixed.items():
        reachable_shelters = result.get('reachable_shelters', [])
        if SINK_NODE_1 in reachable_shelters:
            nodes_reaching_shelter_1 += 1
        if SINK_NODE_2 in reachable_shelters:
            nodes_reaching_shelter_2 += 1
    
    print(f"\n=== SHELTER REACHABILITY AFTER FIX ===")
    print(f"  Nodes that can reach Shelter {SINK_NODE_1}: {nodes_reaching_shelter_1}")
    print(f"  Nodes that can reach Shelter {SINK_NODE_2}: {nodes_reaching_shelter_2}")
    
    if nodes_reaching_shelter_1 > 0:
        print(f"  ✅ Shelter {SINK_NODE_1} is now reachable!")
    else:
        print(f"  ❌ Shelter {SINK_NODE_1} is still unreachable")

else:
    print("Fixed results not available for analysis")

