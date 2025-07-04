import networkx as nx

SINK_NODE_1 = 81  # 避難所1のノードID
SINK_NODE_2 = 213  # 避難所2のノードID

def select_nearest_sink_node(g: nx.Graph, source_node):
    # 各避難所までの距離を計算し、最短の避難所を返す
    distance_to_sink_1 = nx.shortest_path_length(g, source=source_node, target=SINK_NODE_1, weight='length')
    distance_to_sink_2 = nx.shortest_path_length(g, source=source_node, target=SINK_NODE_2, weight='length')
    return SINK_NODE_1 if distance_to_sink_1 < distance_to_sink_2 else SINK_NODE_2