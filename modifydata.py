import pandas as pd
import numpy as np

# Read the existing files
suzu_nodes = pd.read_csv('suzu_nodes.csv')
suzu_edges = pd.read_csv('suzu_edges.csv')
dataset = pd.read_csv('dataset.csv')

print("Original suzu_nodes columns:", suzu_nodes.columns.tolist())
print("Original suzu_edges columns:", suzu_edges.columns.tolist())
print("Target dataset columns:", dataset.columns.tolist())

# Transform suzu_edges to match dataset format
def transform_suzu_edges():
    # Start with the existing suzu_edges
    transformed_edges = suzu_edges.copy()
    
    # Reorder and add missing columns to match dataset format exactly
    target_columns = [
        'u', 'v', 'key', 'osmid', 'highway', 'name', 'oneway', 'ref', 
        'reversed', 'length', 'geometry', 'lanes', 'maxspeed', 'bridge', 
        'tunnel', 'width', 'access'
    ]
    
    # Add missing columns with empty values
    for col in target_columns:
        if col not in transformed_edges.columns:
            if col == 'geometry':
                # Create simple geometry strings based on coordinates from nodes
                transformed_edges[col] = ''
            elif col == 'name':
                # Use the 'name' column if it exists, otherwise empty
                transformed_edges[col] = transformed_edges.get('name', '')
            else:
                transformed_edges[col] = ''
    
    # Handle column mappings and conversions
    if 'est_width' in transformed_edges.columns and 'width' in target_columns:
        # Use est_width for width column
        transformed_edges['width'] = transformed_edges['est_width']
    
    # Ensure proper data types
    transformed_edges['oneway'] = transformed_edges['oneway'].astype(str)
    transformed_edges['reversed'] = transformed_edges['reversed'].astype(str)
    
    # Select only the target columns in the correct order
    transformed_edges = transformed_edges[target_columns]
    
    return transformed_edges

# Transform suzu_nodes to create a nodes file (if needed)
def create_geometry_for_edges():
    """Create simple LINESTRING geometry for edges based on node coordinates"""
    node_coords = {}
    for _, node in suzu_nodes.iterrows():
        node_coords[node['osmid']] = (node['longitude'], node['latitude'])
    
    geometries = []
    for _, edge in suzu_edges.iterrows():
        u_coord = node_coords.get(edge['u'], (0, 0))
        v_coord = node_coords.get(edge['v'], (0, 0))
        
        # Create a simple LINESTRING
        geometry = f"LINESTRING ({u_coord[0]} {u_coord[1]}, {v_coord[0]} {v_coord[1]})"
        geometries.append(geometry)
    
    return geometries

# Apply transformations
print("\nTransforming suzu_edges...")
transformed_edges = transform_suzu_edges()

# Add geometry column
print("Creating geometry data...")
geometries = create_geometry_for_edges()
transformed_edges['geometry'] = geometries

print(f"\nTransformed edges shape: {transformed_edges.shape}")
print("Transformed edges columns:", transformed_edges.columns.tolist())

# Save the transformed files
print("\nSaving transformed files...")
transformed_edges.to_csv('suzu_edges_modified.csv', index=False)

# Also create a properly formatted nodes file
suzu_nodes_modified = suzu_nodes.copy()
# Rename columns to be more consistent
if 'x' in suzu_nodes_modified.columns and 'y' in suzu_nodes_modified.columns:
    suzu_nodes_modified = suzu_nodes_modified.rename(columns={'x': 'longitude', 'y': 'latitude'})

suzu_nodes_modified.to_csv('suzu_nodes_modified.csv', index=False)

print("Files saved as:")
print("- suzu_edges_modified.csv")
print("- suzu_nodes_modified.csv")

# Display sample of transformed data
print("\nSample of transformed edges:")
print(transformed_edges.head(3).to_string())

print(f"\nOriginal edges file had {len(suzu_edges)} rows")
print(f"Transformed edges file has {len(transformed_edges)} rows")