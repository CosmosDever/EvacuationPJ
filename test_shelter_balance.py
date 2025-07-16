#!/usr/bin/env python3
"""
Test the recommended solution for the shelter flow imbalance problem.
This script adjusts shelter capacities and re-runs the evacuation simulation.
"""

import sys
import os
sys.path.append('/home/algorithm/project/EvacuationPJ')

# Import required modules
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import time

def main():
    print("=" * 80)
    print("SHELTER FLOW BALANCE TESTING SCRIPT")
    print("=" * 80)
    
    # Load the data
    print("\n1. Loading node and edge data...")
    try:
        nodes = pd.read_csv('/home/algorithm/project/EvacuationPJ/ginza_nodes.csv')
        edges = pd.read_csv('/home/algorithm/project/EvacuationPJ/ginza_edges.csv')
        print(f"   Loaded {len(nodes)} nodes and {len(edges)} edges")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Check if shelter information is available
    if 'is_shelter' not in nodes.columns:
        print("   No shelter information found. Need to set up shelters first.")
        return
    
    # Show current shelter setup
    shelter_nodes = nodes[nodes['is_shelter'] == True]
    print(f"\n2. Current shelter setup:")
    print(f"   Found {len(shelter_nodes)} shelters")
    
    for idx, row in shelter_nodes.iterrows():
        print(f"   {row.get('shelter_name', f'Shelter {idx}')}: Node {idx}")
        print(f"     Capacity: {row.get('shelter_capacity', 'Unknown'):,}")
        print(f"     Population at node: {row.get('population', 0)}")
    
    # Check if we have evacuation results to compare
    print(f"\n3. Shelter capacity adjustment strategy:")
    print(f"   Based on previous analysis, Shelter 1 received only 1 person")
    print(f"   while Shelter 2 received 958 people (99.9% vs 0.1%)")
    print(f"   ")
    print(f"   Recommended action: Reduce Shelter 2 capacity to force balance")
    
    # Apply the capacity adjustment
    print(f"\n4. Applying capacity adjustments...")
    
    # Find shelter nodes (assume we have Shelter 1 and Shelter 2)
    shelter_1_node = None
    shelter_2_node = None
    
    for idx, row in shelter_nodes.iterrows():
        if row.get('shelter_name') == 'Shelter 1':
            shelter_1_node = idx
        elif row.get('shelter_name') == 'Shelter 2':
            shelter_2_node = idx
    
    if shelter_1_node is None or shelter_2_node is None:
        print("   Error: Could not find both Shelter 1 and Shelter 2 nodes")
        return
    
    # Store original capacities
    original_capacity_1 = nodes.loc[shelter_1_node, 'shelter_capacity']
    original_capacity_2 = nodes.loc[shelter_2_node, 'shelter_capacity']
    
    print(f"   Original Shelter 1 capacity: {original_capacity_1:,}")
    print(f"   Original Shelter 2 capacity: {original_capacity_2:,}")
    
    # Apply recommended changes
    nodes.loc[shelter_1_node, 'shelter_capacity'] = 1000000  # Keep at 1M
    nodes.loc[shelter_2_node, 'shelter_capacity'] = 500000   # Reduce to 500K
    
    print(f"   Adjusted Shelter 1 capacity: {nodes.loc[shelter_1_node, 'shelter_capacity']:,}")
    print(f"   Adjusted Shelter 2 capacity: {nodes.loc[shelter_2_node, 'shelter_capacity']:,}")
    
    total_capacity = nodes.loc[shelter_1_node, 'shelter_capacity'] + nodes.loc[shelter_2_node, 'shelter_capacity']
    total_population = nodes['population'].sum()
    
    print(f"   Total capacity: {total_capacity:,}")
    print(f"   Total population: {total_population:,}")
    print(f"   Capacity ratio: {total_capacity / total_population:.1f}x")
    
    print(f"\n5. Capacity adjustment complete!")
    print(f"   To test this solution, run the evacuation simulation again")
    print(f"   with these adjusted capacities in the Jupyter notebook.")
    
    # Save the adjusted node data for use in notebook
    output_path = '/home/algorithm/project/EvacuationPJ/ginza_nodes_adjusted.csv'
    nodes.to_csv(output_path, index=False)
    print(f"   Saved adjusted node data to: {output_path}")
    
    print("\n" + "=" * 80)
    print("CAPACITY ADJUSTMENT COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run the evacuation simulation in the Jupyter notebook")
    print("2. Compare the shelter flow distribution results")
    print("3. Verify that the balance has improved")

if __name__ == "__main__":
    main()
