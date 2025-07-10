import networkx as nx

SINK_NODE_1 = 1281123728
SINK_NODE_2 = 1281090837
SINK_NODE_3 = 1281108573

def select_nearest_sink_node(g: nx.Graph, source_node):
    # 各避難所までの距離を計算し、最短の避難所を返す
    distance_to_sink_1 = nx.shortest_path_length(g, source=source_node, target=SINK_NODE_1, weight='length')
    distance_to_sink_2 = nx.shortest_path_length(g, source=source_node, target=SINK_NODE_2, weight='length')
    distance_to_sink_3 = nx.shortest_path_length(g, source=source_node, target=SINK_NODE_3, weight='length')
    return SINK_NODE_1 if distance_to_sink_1 < distance_to_sink_2 else SINK_NODE_2