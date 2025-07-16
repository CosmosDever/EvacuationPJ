#!/usr/bin/env python3
"""
Investigate why no one is going to Shelter 1 in the evacuation simulation.
This script analyzes the flow results to understand shelter usage patterns.
"""

import pandas as pd
import numpy as np
import pickle
import sys
from collections import defaultdict

def analyze_shelter_flow(notebook_path):
    """
    Analyze the flow results to understand why Shelter 1 is not being used.
    """
    print("="*80)
    print("SHELTER FLOW INVESTIGATION")
    print("="*80)
    
    # We'll need to extract evacuation_result from the notebook
    # For now, let's create a function that can be called from the notebook
    
    print("\nThis script needs to be run from within the Jupyter notebook context")
    print("to access the evacuation_result variable.")
    
    print("\nTo investigate the Shelter 1 usage issue, please run the following")
    print("analysis code in your notebook:")
    
    analysis_code = '''
def investigate_shelter_flow_detailed(evacuation_result, nodes):
    """Detailed investigation of flow to each shelter."""
    if evacuation_result is None:
        print("No evacuation result available")
        return
    
    print("="*80)
    print("SHELTER FLOW DETAILED INVESTIGATION")
    print("="*80)
    
    flow_dict = evacuation_result['flow_dict']
    ten = evacuation_result['time_expanded_network']
    
    # Get shelter information
    shelter_nodes = nodes[nodes['is_shelter']].copy()
    print(f"\\nShelter Information:")
    for idx, row in shelter_nodes.iterrows():
        print(f"  {row['shelter_name']}: Node {idx}")
        print(f"    Capacity: {row['shelter_capacity']:,}")
        print(f"    Coordinates: ({row.geometry.x:.6f}, {row.geometry.y:.6f})")
        print()
    
    # Analyze flow to each shelter across all time steps
    shelter_flow_analysis = {}
    
    for shelter_idx, shelter_row in shelter_nodes.iterrows():
        shelter_name = shelter_row['shelter_name']
        shelter_flow_analysis[shelter_name] = {
            'node_id': shelter_idx,
            'total_inflow': 0,
            'time_step_flows': {},
            'connected_to_sink': False,
            'sink_connections': []
        }
        
        # Check all time steps for this shelter
        for t in range(ten.time_steps):
            if (shelter_idx, t) in ten.node_mapping:
                time_shelter_node = ten.node_mapping[(shelter_idx, t)]
                
                # Check inflow to this shelter at this time step
                inflow = 0
                for source_node, targets in flow_dict.items():
                    if time_shelter_node in targets and targets[time_shelter_node] > 0:
                        inflow += targets[time_shelter_node]
                
                if inflow > 0:
                    shelter_flow_analysis[shelter_name]['time_step_flows'][t] = inflow
                    shelter_flow_analysis[shelter_name]['total_inflow'] += inflow
                
                # Check if this shelter node connects to sink
                if time_shelter_node in flow_dict:
                    sink_flow = flow_dict[time_shelter_node].get('super_sink', 0)
                    if sink_flow > 0:
                        shelter_flow_analysis[shelter_name]['connected_to_sink'] = True
                        shelter_flow_analysis[shelter_name]['sink_connections'].append({
                            'time_step': t,
                            'flow_to_sink': sink_flow
                        })
    
    # Print detailed analysis
    print("\\nDetailed Flow Analysis by Shelter:")
    print("-" * 50)
    
    for shelter_name, analysis in shelter_flow_analysis.items():
        print(f"\\n{shelter_name} (Node {analysis['node_id']}):")
        print(f"  Total Inflow: {analysis['total_inflow']}")
        print(f"  Connected to Sink: {analysis['connected_to_sink']}")
        
        if analysis['time_step_flows']:
            print(f"  Time Steps with Inflow: {len(analysis['time_step_flows'])}")
            print(f"  Flow Timeline (first 10 time steps):")
            for t in sorted(analysis['time_step_flows'].keys())[:10]:
                print(f"    Time {t}: {analysis['time_step_flows'][t]} people")
        else:
            print(f"  ⚠️  NO INFLOW DETECTED - This shelter received no evacuees!")
        
        if analysis['sink_connections']:
            print(f"  Sink Connections: {len(analysis['sink_connections'])}")
            total_sink_flow = sum(conn['flow_to_sink'] for conn in analysis['sink_connections'])
            print(f"  Total Flow to Sink: {total_sink_flow}")
        else:
            print(f"  ⚠️  NO SINK CONNECTION - This shelter is not evacuating people!")
    
    # Check network connectivity to Shelter 1
    print("\\n" + "="*80)
    print("NETWORK CONNECTIVITY ANALYSIS FOR SHELTER 1")
    print("="*80)
    
    shelter_1_node = None
    for shelter_idx, shelter_row in shelter_nodes.iterrows():
        if shelter_row['shelter_name'] == 'Shelter 1':
            shelter_1_node = shelter_idx
            break
    
    if shelter_1_node is not None:
        print(f"\\nAnalyzing connectivity to Shelter 1 (Node {shelter_1_node})...")
        
        # Check if there are any edges leading to Shelter 1 in the time-expanded network
        shelter_1_predecessors = 0
        shelter_1_connections = []
        
        for t in range(min(10, ten.time_steps)):  # Check first 10 time steps
            if (shelter_1_node, t) in ten.node_mapping:
                time_shelter_node = ten.node_mapping[(shelter_1_node, t)]
                
                # Count predecessors in time-expanded network
                if time_shelter_node in ten.time_expanded_graph:
                    predecessors = list(ten.time_expanded_graph.predecessors(time_shelter_node))
                    if predecessors:
                        shelter_1_predecessors += len(predecessors)
                        shelter_1_connections.append({
                            'time_step': t,
                            'predecessors': len(predecessors),
                            'first_few_predecessors': predecessors[:5]
                        })
        
        print(f"  Total predecessor connections across time steps: {shelter_1_predecessors}")
        print(f"  Time steps with connections: {len(shelter_1_connections)}")
        
        if shelter_1_connections:
            print(f"  Connection details (first 5 time steps):")
            for conn in shelter_1_connections[:5]:
                print(f"    Time {conn['time_step']}: {conn['predecessors']} predecessors")
        else:
            print(f"  ⚠️  NO PREDECESSORS FOUND - Shelter 1 may not be reachable!")
    
    # Check population distribution around shelters
    print("\\n" + "="*80)
    print("POPULATION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"\\nPopulation near shelters:")
    for shelter_idx, shelter_row in shelter_nodes.iterrows():
        shelter_name = shelter_row['shelter_name']
        shelter_pop = shelter_row['population']
        print(f"  {shelter_name}: {shelter_pop} people at shelter node itself")
        
        # Find nearby nodes with population
        shelter_coords = (shelter_row.geometry.x, shelter_row.geometry.y)
        nearby_pop = 0
        nearby_nodes = 0
        
        for node_idx, node_row in nodes.iterrows():
            if node_row['population'] > 0:
                node_coords = (node_row.geometry.x, node_row.geometry.y)
                distance = ((shelter_coords[0] - node_coords[0])**2 + 
                           (shelter_coords[1] - node_coords[1])**2)**0.5
                
                if distance < 0.01:  # Within ~1km (very rough approximation)
                    nearby_pop += node_row['population']
                    nearby_nodes += 1
        
        print(f"    Population within 0.01° radius: {nearby_pop} people in {nearby_nodes} nodes")
    
    return shelter_flow_analysis

# Run the investigation
if 'evacuation_result' in locals() and 'nodes' in locals():
    shelter_analysis = investigate_shelter_flow_detailed(evacuation_result, nodes)
else:
    print("Error: evacuation_result or nodes not found in current scope")
'''
    
    print(analysis_code)
    
    return analysis_code

if __name__ == "__main__":
    # If run as script, just display the analysis code
    analysis_code = analyze_shelter_flow("/home/algorithm/project/EvacuationPJ/Test_on_Ahi_Ginza.ipynb")
