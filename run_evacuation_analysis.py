#!/usr/bin/env python3
"""
Enhanced Time-Expanded Network Evacuation Analysis
This script implements max flow analysis and time-expanded network simulation
with animation capabilities for evacuation planning.
"""

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
import json
from datetime import datetime

# Animation and visualization imports
try:
    from matplotlib.animation import FuncAnimation
    from IPython.display import display, clear_output
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    animation_available = True
except ImportError as e:
    print(f"Animation libraries not available: {e}")
    animation_available = False

# Parallel processing optimization
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Network analysis
from networkx.algorithms.flow import shortest_augmenting_path
import re

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

def load_data():
    """Load and process the evacuation network data"""
    print("Loading data from dataset.csv...")
    try:
        edges_df = pd.read_csv('/home/algorithm/project/EvacuationPJ/dataset.csv')
        print(f"Successfully loaded {len(edges_df)} edges from dataset.csv")
        return edges_df
    except Exception as e:
        print(f"Error loading dataset.csv: {e}")
        return None

def create_nodes_with_population(edges_df):
    """Create nodes dataframe with population data"""
    if edges_df is None:
        return None
        
    # Create nodes from unique source and target nodes
    unique_nodes = pd.unique(edges_df[['u', 'v']].values.ravel('K'))
    
    # Create nodes DataFrame with basic geometry (simplified coordinates)
    nodes_data = []
    for i, node_id in enumerate(unique_nodes):
        nodes_data.append({'node_id': node_id})
    
    nodes = pd.DataFrame(nodes_data).set_index('node_id')
    
    # Add random population to each node
    np.random.seed(42)  # For reproducible results
    num_nodes = len(nodes)
    
    # Generate random population for each node using different distribution strategies
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
    
    return nodes

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

def create_networkx_graph(edges_df, node_coordinates):
    """Create NetworkX graph for network analysis"""
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
    
    return G

class MaxFlowAnalyzer:
    """Analyze maximum flow from all nodes to shelter nodes"""
    
    def __init__(self, graph, shelter_nodes):
        self.graph = graph.copy()
        self.shelter_nodes = shelter_nodes
        self.results = {}
        
        print(f"MaxFlowAnalyzer initialized with {len(shelter_nodes)} shelters")
        
    def calculate_edge_capacity(self, u, v, edge_data):
        """Calculate edge capacity based on road type and characteristics"""
        base_length = edge_data.get('length', 100)
        highway_type = edge_data.get('highway', 'residential')
        
        # Capacity multipliers based on road type
        multipliers = {
            'motorway': 10,
            'trunk': 8,
            'primary': 6,
            'secondary': 4,
            'tertiary': 3,
            'residential': 2,
            'service': 1,
            'unclassified': 1
        }
        
        multiplier = multipliers.get(highway_type, 2)
        base_capacity = max(1, int(base_length * 0.01))  # 1% of length as base
        
        return base_capacity * multiplier
    
    @monitor_performance
    def analyze_max_flows(self):
        """Calculate maximum flow from each node to shelter nodes"""
        print("\n🔄 Calculating maximum flows from all nodes to shelters...")
        
        # Add capacities to edges
        for u, v, data in self.graph.edges(data=True):
            capacity = self.calculate_edge_capacity(u, v, data)
            self.graph[u][v]['capacity'] = capacity
        
        # Create supersink connected to all shelter nodes
        supersink = "SUPERSINK_TEMP"
        self.graph.add_node(supersink, is_supersink=True)
        
        for shelter in self.shelter_nodes:
            if shelter in self.graph.nodes():
                self.graph.add_edge(shelter, supersink, capacity=float('inf'))
        
        # Calculate max flow from each node
        successful_calculations = 0
        total_capacity = 0
        error_nodes = []
        
        for source_node in self.graph.nodes():
            if source_node == supersink or source_node in self.shelter_nodes:
                continue
                
            try:
                from networkx.algorithms.flow import maximum_flow
                max_flow_value, _ = maximum_flow(
                    self.graph, source_node, supersink, 
                    capacity='capacity', flow_func=shortest_augmenting_path
                )
                
                self.results[source_node] = {
                    'max_flow': max_flow_value,
                    'success': True
                }
                
                successful_calculations += 1
                total_capacity += max_flow_value
                
            except Exception as e:
                self.results[source_node] = {
                    'max_flow': 0,
                    'success': False,
                    'error': str(e)
                }
                error_nodes.append(source_node)
        
        # Remove temporary supersink
        self.graph.remove_node(supersink)
        
        # Print results
        total_nodes = len([n for n in self.graph.nodes() if n not in self.shelter_nodes])
        success_rate = (successful_calculations / total_nodes) * 100
        
        print(f"\n📊 Max Flow Analysis Results:")
        print(f"   Total nodes analyzed: {total_nodes}")
        print(f"   Successful calculations: {successful_calculations}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total flow capacity: {total_capacity:,} units")
        print(f"   Error nodes: {len(error_nodes)}")
        
        if error_nodes:
            print(f"   Error node examples: {error_nodes[:5]}")
        
        return self.results

class TimeExpandedEvacuationNetwork:
    """Enhanced Time-Expanded Network for evacuation simulation with animation"""
    
    def __init__(self, base_graph, nodes_df, max_flow_results, shelter_nodes, 
                 time_horizon=90, time_step=3):
        self.base_graph = base_graph
        self.nodes_df = nodes_df
        self.max_flow_results = max_flow_results
        self.shelter_nodes = shelter_nodes
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.time_periods = list(range(0, time_horizon + 1, time_step))
        
        # Initialize containers for results
        self.time_expanded_graph = None
        self.flow_solution = None
        self.evacuation_state = {}
        self.animation_data = []
        
        print(f"🚀 Enhanced Time-Expanded Network initialized:")
        print(f"   Time horizon: {time_horizon} minutes")
        print(f"   Time step: {time_step} minute(s)")
        print(f"   Time periods: {len(self.time_periods)}")
        print(f"   Base nodes: {len(base_graph.nodes())}")
        print(f"   Shelters: {len(shelter_nodes)}")
        
    @monitor_performance
    def create_time_expanded_graph(self):
        """Create the time-expanded network graph"""
        print("\n📊 Creating time-expanded network...")
        
        G_time = nx.DiGraph()
        
        # Add time-indexed nodes
        for t in self.time_periods:
            for node in self.base_graph.nodes():
                time_node = f"{node}_t{t}"
                node_attrs = {
                    'original_node': node,
                    'time_period': t,
                    'is_shelter': node in self.shelter_nodes
                }
                
                # Add node attributes from original graph and dataframe
                if node in self.base_graph.nodes():
                    node_attrs.update(self.base_graph.nodes[node])
                
                if node in self.nodes_df.index:
                    node_attrs.update({
                        'population': self.nodes_df.loc[node, 'population'],
                        'node_type': self.nodes_df.loc[node, 'node_type']
                    })
                    
                G_time.add_node(time_node, **node_attrs)
        
        print(f"   ✓ Added {G_time.number_of_nodes():,} time-indexed nodes")
        
        # Add spatial edges (within same time period)
        spatial_edges = 0
        for t in self.time_periods:
            for u, v, data in self.base_graph.edges(data=True):
                time_u = f"{u}_t{t}"
                time_v = f"{v}_t{t}"
                
                if time_u in G_time.nodes() and time_v in G_time.nodes():
                    # Calculate travel time
                    travel_time = data.get('travel_time', data.get('length', 100) / 30)
                    
                    # Add edge if travel can be completed within time step
                    if travel_time <= self.time_step:
                        capacity = self._calculate_edge_capacity(u, v, data)
                        
                        G_time.add_edge(time_u, time_v,
                                      edge_type='spatial',
                                      capacity=capacity,
                                      travel_time=travel_time,
                                      original_edge=(u, v))
                        spatial_edges += 1
        
        print(f"   ✓ Added {spatial_edges:,} spatial edges")
        
        # Add temporal edges (waiting at same location)
        temporal_edges = 0
        for i in range(len(self.time_periods) - 1):
            t_current = self.time_periods[i]
            t_next = self.time_periods[i + 1]
            
            for node in self.base_graph.nodes():
                time_current = f"{node}_t{t_current}"
                time_next = f"{node}_t{t_next}"
                
                if time_current in G_time.nodes() and time_next in G_time.nodes():
                    waiting_capacity = self._get_waiting_capacity(node)
                    
                    G_time.add_edge(time_current, time_next,
                                  edge_type='temporal',
                                  capacity=waiting_capacity,
                                  travel_time=self.time_step)
                    temporal_edges += 1
        
        print(f"   ✓ Added {temporal_edges:,} temporal edges")
        
        # Add super source and sink
        self._add_super_nodes(G_time)
        
        self.time_expanded_graph = G_time
        print(f"\n🎯 Time-expanded graph created successfully!")
        print(f"   Total nodes: {G_time.number_of_nodes():,}")
        print(f"   Total edges: {G_time.number_of_edges():,}")
        
        return G_time
    
    def _calculate_edge_capacity(self, u, v, edge_data):
        """Calculate edge capacity based on road type and max flow results"""
        base_capacity = edge_data.get('capacity', 10)
        
        # Adjust based on max flow results
        u_max_flow = self.max_flow_results.get(u, {}).get('max_flow', 0)
        v_max_flow = self.max_flow_results.get(v, {}).get('max_flow', 0)
        
        # Use average max flow as capacity multiplier
        avg_flow = (u_max_flow + v_max_flow) / 2
        capacity_multiplier = max(0.5, min(2.0, avg_flow / 20))  # Scale between 0.5x and 2x
        
        return max(1, int(base_capacity * capacity_multiplier))
    
    def _get_waiting_capacity(self, node):
        """Get waiting capacity for a node"""
        if node in self.max_flow_results:
            max_flow = self.max_flow_results[node].get('max_flow', 0)
            return max(5, int(max_flow * 0.3))  # 30% of max flow as waiting capacity
        return 5
    
    def _add_super_nodes(self, G_time):
        """Add super source and super sink"""
        super_source = "SUPER_SOURCE"
        super_sink = "SUPER_SINK"
        
        G_time.add_node(super_source, node_type='super_source')
        G_time.add_node(super_sink, node_type='super_sink')
        
        # Connect super source to all nodes at t=0 with population as capacity
        for node in self.base_graph.nodes():
            if node in self.nodes_df.index:
                population = self.nodes_df.loc[node, 'population']
                time_node = f"{node}_t{self.time_periods[0]}"
                
                if time_node in G_time.nodes():
                    G_time.add_edge(super_source, time_node,
                                  edge_type='source',
                                  capacity=population,
                                  travel_time=0)
        
        # Connect all shelter nodes at all times to super sink
        for t in self.time_periods:
            for shelter in self.shelter_nodes:
                time_shelter = f"{shelter}_t{t}"
                if time_shelter in G_time.nodes():
                    G_time.add_edge(time_shelter, super_sink,
                                  edge_type='sink',
                                  capacity=float('inf'),
                                  travel_time=0)
        
        print(f"   ✓ Added super source and super sink")
    
    @monitor_performance
    def solve_max_flow(self):
        """Solve maximum flow on time-expanded network"""
        if self.time_expanded_graph is None:
            print("❌ Error: Time-expanded graph not created!")
            return None
        
        print("\n🔄 Solving maximum flow on time-expanded network...")
        
        try:
            from networkx.algorithms.flow import maximum_flow
            
            flow_value, flow_dict = maximum_flow(
                self.time_expanded_graph,
                "SUPER_SOURCE",
                "SUPER_SINK",
                capacity='capacity',
                flow_func=shortest_augmenting_path
            )
            
            self.flow_solution = {
                'flow_value': flow_value,
                'flow_dict': flow_dict
            }
            
            total_population = self.nodes_df['population'].sum()
            efficiency = (flow_value / total_population) * 100
            
            print(f"\n📈 Flow solution computed:")
            print(f"   Maximum flow: {flow_value:,}")
            print(f"   Total population: {total_population:,}")
            print(f"   Evacuation efficiency: {efficiency:.1f}%")
            
            return flow_value, flow_dict
            
        except Exception as e:
            print(f"❌ Error solving max flow: {e}")
            return None, None
    
    def extract_evacuation_dynamics(self):
        """Extract detailed evacuation dynamics from flow solution"""
        if not self.flow_solution:
            print("❌ Error: No flow solution available!")
            return None
        
        print("\n📊 Extracting evacuation dynamics...")
        
        flow_dict = self.flow_solution['flow_dict']
        
        # Initialize evacuation state tracking
        time_series_data = []
        
        # Process flows for each time period
        for t in self.time_periods:
            period_data = {
                'time': t,
                'evacuating': 0,
                'evacuated': 0,
                'waiting': 0,
                'node_details': {},
                'edge_details': {}
            }
            
            # Track node states
            for node in self.base_graph.nodes():
                time_node = f"{node}_t{t}"
                
                if time_node in flow_dict:
                    outflow = sum(flow_dict[time_node].values())
                    inflow = sum([flows.get(time_node, 0) for flows in flow_dict.values()])
                    
                    # Determine node state
                    if node in self.shelter_nodes:
                        state = 'shelter'
                    elif outflow > 0:
                        state = 'evacuating'
                    elif inflow > outflow:
                        state = 'waiting'
                    else:
                        state = 'stable'
                    
                    period_data['node_details'][node] = {
                        'state': state,
                        'inflow': inflow,
                        'outflow': outflow,
                        'population': self.nodes_df.loc[node, 'population'] if node in self.nodes_df.index else 0
                    }
                    
                    # Update counters
                    if state == 'evacuating':
                        period_data['evacuating'] += outflow
                    elif state == 'waiting':
                        period_data['waiting'] += (inflow - outflow)
            
            # Calculate evacuated (flow to shelters)
            for shelter in self.shelter_nodes:
                shelter_time = f"{shelter}_t{t}"
                if shelter_time in flow_dict:
                    for sink_node, flow in flow_dict[shelter_time].items():
                        if sink_node == "SUPER_SINK":
                            period_data['evacuated'] += flow
            
            time_series_data.append(period_data)
        
        self.evacuation_state = {
            'time_series': time_series_data,
            'summary': self._create_evacuation_summary(time_series_data)
        }
        
        print(f"   ✓ Extracted dynamics for {len(time_series_data)} time periods")
        return self.evacuation_state
    
    def _create_evacuation_summary(self, time_series_data):
        """Create summary statistics from time series data"""
        total_evacuated = sum([period['evacuated'] for period in time_series_data])
        peak_evacuating = max([period['evacuating'] for period in time_series_data])
        total_population = self.nodes_df['population'].sum()
        
        # Find evacuation completion time
        completion_time = None
        for period in reversed(time_series_data):
            if period['evacuating'] > 0:
                completion_time = period['time']
                break
        
        return {
            'total_evacuated': total_evacuated,
            'peak_evacuating_rate': peak_evacuating,
            'completion_time': completion_time,
            'evacuation_efficiency': (total_evacuated / total_population) * 100,
            'total_population': total_population
        }
    
    def generate_report(self, save_path=None):
        """Generate comprehensive analysis report"""
        if not self.evacuation_state:
            print("❌ Error: No evacuation dynamics available!")
            return None
        
        print("\n📋 Generating analysis report...")
        
        summary = self.evacuation_state['summary']
        time_series = self.evacuation_state['time_series']
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "network_configuration": {
                "time_horizon": self.time_horizon,
                "time_step": self.time_step,
                "total_periods": len(self.time_periods),
                "base_nodes": len(self.base_graph.nodes()),
                "time_expanded_nodes": self.time_expanded_graph.number_of_nodes() if self.time_expanded_graph else 0,
                "time_expanded_edges": self.time_expanded_graph.number_of_edges() if self.time_expanded_graph else 0
            },
            "evacuation_results": {
                "total_population": summary['total_population'],
                "total_evacuated": summary['total_evacuated'],
                "evacuation_efficiency": summary['evacuation_efficiency'],
                "peak_evacuation_rate": summary['peak_evacuating_rate'],
                "completion_time": summary['completion_time']
            },
            "performance_metrics": {
                "average_evacuation_rate": np.mean([data['evacuating'] for data in time_series]),
                "peak_waiting": max([data['waiting'] for data in time_series]),
                "utilization_rate": summary['total_evacuated'] / (summary['total_population'] * len(self.time_periods))
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"   ✓ Report saved to {save_path}")
        
        # Print summary
        print(f"\n📊 EVACUATION ANALYSIS REPORT")
        print(f"=" * 50)
        print(f"Total Population: {summary['total_population']:,}")
        print(f"Total Evacuated: {summary['total_evacuated']:,}")
        print(f"Evacuation Efficiency: {summary['evacuation_efficiency']:.1f}%")
        print(f"Peak Evacuation Rate: {summary['peak_evacuating_rate']:,} people/min")
        print(f"Completion Time: {summary['completion_time']} minutes")
        print(f"=" * 50)
        
        return report

def create_static_visualization(evacuation_state, save_path=None):
    """Create static visualization plots"""
    if not evacuation_state:
        print("❌ No evacuation state data available!")
        return
    
    print("\n📊 Creating static visualizations...")
    
    time_series = evacuation_state['time_series']
    
    # Prepare data
    times = [data['time'] for data in time_series]
    evacuating = [data['evacuating'] for data in time_series]
    evacuated = [data['evacuated'] for data in time_series]
    waiting = [data['waiting'] for data in time_series]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Evacuation progress over time
    axes[0,0].plot(times, evacuating, 'r-', linewidth=2, label='Currently Evacuating')
    axes[0,0].plot(times, evacuated, 'g-', linewidth=2, label='Total Evacuated')
    axes[0,0].plot(times, waiting, 'orange', linewidth=2, label='Waiting')
    axes[0,0].set_xlabel('Time (minutes)')
    axes[0,0].set_ylabel('Number of People')
    axes[0,0].set_title('Evacuation Progress Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Evacuation efficiency over time
    total_population = evacuation_state['summary']['total_population']
    cumulative_evacuated = np.cumsum(evacuated)
    efficiency_over_time = (cumulative_evacuated / total_population) * 100
    axes[0,1].plot(times, efficiency_over_time, 'b-', linewidth=2)
    axes[0,1].set_xlabel('Time (minutes)')
    axes[0,1].set_ylabel('Evacuation Efficiency (%)')
    axes[0,1].set_title('Cumulative Evacuation Efficiency')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Flow rates
    flow_rates = [0] + [evacuated[i] - evacuated[i-1] for i in range(1, len(evacuated))]
    axes[1,0].plot(times, flow_rates, 'purple', linewidth=2)
    axes[1,0].set_xlabel('Time (minutes)')
    axes[1,0].set_ylabel('People per Period')
    axes[1,0].set_title('Evacuation Flow Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    summary = evacuation_state['summary']
    metrics = ['Total\nEvacuated', 'Peak\nRate', 'Efficiency\n(%)']
    values = [summary['total_evacuated'], summary['peak_evacuating_rate'], summary['evacuation_efficiency']]
    
    bars = axes[1,1].bar(metrics, values, color=['green', 'blue', 'orange'], alpha=0.7)
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Summary Statistics')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                      f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Visualization saved to {save_path}")
    
    plt.show()
    return fig

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Time-Expanded Network Evacuation Analysis")
    print("=" * 70)
    
    # Define shelter nodes
    SINK_NODE_1 = 31253600 
    SINK_NODE_2 = 5110476586
    shelter_nodes = [SINK_NODE_1, SINK_NODE_2]
    
    # Step 1: Load and process data
    print("\n📁 STEP 1: Loading and processing data...")
    edges_df = load_data()
    if edges_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    nodes = create_nodes_with_population(edges_df)
    if nodes is None:
        print("❌ Failed to create nodes. Exiting.")
        return
    
    print(f"✅ Data loaded successfully:")
    print(f"   Nodes: {len(nodes):,}")
    print(f"   Edges: {len(edges_df):,}")
    print(f"   Total population: {nodes['population'].sum():,}")
    
    # Step 2: Extract coordinates and create graph
    print("\n🗺️  STEP 2: Creating network graph...")
    node_coordinates = extract_node_coordinates(edges_df)
    
    # Add coordinates to nodes dataframe
    nodes['lat'] = nodes.index.map(lambda x: node_coordinates.get(x, [None, None])[0])
    nodes['lon'] = nodes.index.map(lambda x: node_coordinates.get(x, [None, None])[1])
    
    # Create NetworkX graph
    road_network_graph = create_networkx_graph(edges_df, node_coordinates)
    
    # Step 3: Max flow analysis
    print("\n🔄 STEP 3: Max flow analysis...")
    analyzer = MaxFlowAnalyzer(road_network_graph, shelter_nodes)
    max_flow_results = analyzer.analyze_max_flows()
    
    # Step 4: Time-expanded network analysis
    print("\n⏰ STEP 4: Time-expanded network analysis...")
    time_horizon = 90  # 1.5 hours simulation
    time_step = 3      # 3-minute intervals
    
    ten = TimeExpandedEvacuationNetwork(
        base_graph=road_network_graph,
        nodes_df=nodes,
        max_flow_results=max_flow_results,
        shelter_nodes=shelter_nodes,
        time_horizon=time_horizon,
        time_step=time_step
    )
    
    # Create time-expanded graph
    time_expanded_graph = ten.create_time_expanded_graph()
    
    # Solve maximum flow
    flow_value, flow_dict = ten.solve_max_flow()
    
    if flow_value is None:
        print("❌ Failed to solve max flow. Exiting.")
        return
    
    # Extract evacuation dynamics
    evacuation_dynamics = ten.extract_evacuation_dynamics()
    
    # Generate report
    report = ten.generate_report(save_path="evacuation_analysis_report.json")
    
    # Step 5: Create visualizations
    print("\n📊 STEP 5: Creating visualizations...")
    create_static_visualization(evacuation_dynamics, save_path="evacuation_analysis.png")
    
    print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Report saved to: evacuation_analysis_report.json")
    print(f"📊 Visualization saved to: evacuation_analysis.png")
    
    return {
        'network': ten,
        'graph': time_expanded_graph,
        'flow_value': flow_value,
        'dynamics': evacuation_dynamics,
        'report': report
    }

if __name__ == "__main__":
    results = main()
