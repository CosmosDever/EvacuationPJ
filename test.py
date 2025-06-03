import osmnx as ox

import networkx as nx

import geopandas as gpd

import matplotlib.pyplot as plt

import pandas as pd

place_name = "Suzu , Japan"

graph = ox.graph_from_place(place_name, network_type='drive')

fig, ax = ox.plot_graph(graph)
