from typing import Literal
import networkx as nx
import pandas as pd
from decimal import *
from math import *
from select_node import select_nearest_sink_node
import rand
from compute import compute_travel_time

SINK_NODE_1 = 81  # 避難所1のノードID
SINK_NODE_2 = 213  # 避難所2のノードID

Model = Literal['bpr', 'greenshields', 'exponential']
Weight = Literal['length', 'travel_time']
Order = Literal['near', 'far', 'random']

def default_arrival_callback(i: int, simulations: int, fin_count: int, ctime: int):
    print(f"避難者{i+1}は避難所に到達しました, 残り人数: {simulations - fin_count} 現在時刻: {ctime}")

def simulate(
        csv_file: str='data.csv', 
        seed: int=1, 
        model: Model='exponential',
        weight: Weight='length',
        order: Order='near',
        simulations: int=3000,
        max_time: int=inf,
        human_speeds: list=[1.25],
        wait_times: list=[0],
        default_human_speed: float=1.25,
        time_interval: float=1,
        arrival_callback=default_arrival_callback,
        v_min: float=0.03,
        exponential_alpha: float=2.0,
        bpr_alpha: float=1.919,
        bpr_beta: float=6.9373
    ) -> pd.DataFrame:
    """
    シミュレーションの実行

    csv_file: 道路情報が記述されたCSVファイル
    seed: 乱数のシード
    model: モデルの指定（'bpr' or 'greenshields' or 'exponential'）
    weight: エッジの重みの指定（'length' or 'travel_time'）
    order: 避難者の順序（'near' or 'far' or 'random'）
    simulations: シミュレーションの回数
    max_time: シミュレーション最大の時間（秒）
    human_speed: 人の速度（m/s）
    time_interval: 時間間隔
    arrival_callback: 避難所に到達した時のコールバック関数

    Returns 避難時間のデータフレーム
    """
    rnd = rand.DeterministicRandom(seed)
    # 新しいグラフの初期化
    graph_with_time = nx.DiGraph()
    # CSVから道路情報を読み込み、重み付きエッジを追加
    edges_df = pd.read_csv(csv_file)
    # ノードIDを取得して1次元配列に変換後、重複を削除
    unique_nodes = pd.unique(edges_df[['u', 'v']].values.ravel('K'))
    node_id_dict = {node: idx + 1 for idx, node in enumerate(unique_nodes)}
    edges_df['travel_time'] = (edges_df['length'] / default_human_speed).round(4)
    edges_df = edges_df[['u', 'v', 'length', 'travel_time', 'highway']]
    # 道路タイプごとのキャパシティ係数
    capacity_coefficients = {'trunk': 10,'primary': 10,'tertiary': 10,'unclassified': 5,'secondary': 5,'footway': 2,'foot': 2,'residential': 2}

    # 移動時間と容量を持たせたグラフ作成
    capacity_dict = {}
    init_time_dict = {}
    current_travel_time_dict = {}
    now_dict = {}
    pass_count_dict = {}
    n_di_dict = {}
    for _, row in edges_df.iterrows():
        source_node = node_id_dict[row['u']]
        target_node = node_id_dict[row['v']]
        length = row['length']
        travel_time = row['travel_time']
        highway_type = row['highway']
        coefficient = capacity_coefficients.get(highway_type, 1)
        capacity = round(coefficient * length, 2)
        # 重み付きエッジを追加
        if weight == "length":
            graph_with_time.add_edge(source_node, target_node, weight=length)
        elif weight == "travel_time":
            graph_with_time.add_edge(source_node, target_node, weight=travel_time)
        source_target = (source_node, target_node)
        # キャパシティと移動時間を保存
        capacity_dict[source_target] = capacity
        init_time_dict[source_target] = travel_time
        current_travel_time_dict[source_target] = travel_time
        now_dict[source_target] = 0
        pass_count_dict[source_target] = 0
        n_di_dict[source_target] = length
    
    def dijkstra_with_nx_dijkstra_path(graph):
        paths = []
        error_count = 0
        goal_select = 0
        i = 0

        node_choice_list = []
        while(i != 500):
            try:
                start = rnd.choice(list(graph.nodes))
                if start in (SINK_NODE_1, SINK_NODE_2):
                    continue
                path = nx.dijkstra_path(graph, source=start, target=select_nearest_sink_node(graph, start), weight='weight')
                length = 0
                for j in range(len(path)-1):
                    source_target = (path[j], path[j+1])
                    length += n_di_dict[source_target]
                node_choice_list.append((length, path))
                i += 1
            except nx.NetworkXNoPath:
                continue
    
        i = 0
        while(i != simulations):
            node = node_choice_list[i % 500]
            paths.append((node[0], node[1], i))
            i += 1

        return paths, error_count, goal_select

    paths, no_path_count, from_goal_count = dijkstra_with_nx_dijkstra_path(graph_with_time)

    # paths.sort(key=lambda x: x[0])
    c_paths = rnd.shuffle(paths)
    # 人数
    path_length = len(paths)
    human_speeds.sort()
    speed_count = len(human_speeds)
    sp_per_path = ceil(path_length / speed_count)
    speed_list = []

    speed_map = {}
    length_map = {}
    for i, path in enumerate(c_paths):
        sp = i // sp_per_path
        speed_list.append((human_speeds[sp]))
        speed_map[path[2]] = human_speeds[sp]
        length_map[path[2]] = path[0]

    if (order == 'near'):
        paths.sort(key=lambda x: x[0])
    elif (order == 'far'):
        paths.sort(key=lambda x: x[0], reverse=True)
    elif (order == 'random'):
        paths = rnd.shuffle(paths)

    wait_times.sort()
    wait_count = len(wait_times)
    wait_per_path = ceil(path_length / wait_count)
    wait_list = []
    wait_map = {}
    for i, path in enumerate(c_paths):
        wait = i // wait_per_path
        wait_list.append(wait_times[wait])
        wait_map[path[2]] = wait_times[wait]
    
    # 避難者の状態
    yet = []
    # 避難完了した避難者
    fin_evacuate = {}
    # 避難完了した避難者の数
    fin_count = 0

    for i, value in enumerate(paths):
        init_start_node = value[1][0]
        # 現在、最も移動時間が短い経路を選択
        if weight == "travel_time":
            path = nx.dijkstra_path(
                graph_with_time, 
                source=init_start_node,
                target=select_nearest_sink_node(graph_with_time, init_start_node),
                weight='weight'
            )
        elif weight == "length":
            path = value[1]
        # 現在、最も移動時間が短い経路を通る場合の次のノード
        init_target_node = path[1]
        source_target = (init_start_node, init_target_node)
        travel_time = current_travel_time_dict[source_target]
        current = {
            "node_to_node": source_target,
            "from_index": 0,
            "need_time": travel_time,
            "index": 0
        }
        yet.append(current)
        fin_evacuate[i] = False   
        # そこの道を通る人数をカウント  
        pass_count_dict[source_target] += 1
        # 移動時間の算出
        after_bpr_time = compute_travel_time(
            n_di_dict[source_target], 
            pass_count_dict[source_target], 
            capacity_dict[source_target], 
            model=model,
            V_DEFAULT=speed_list[i],
            V_MIN=v_min,
            EXPONENTIAL_ALPHA=exponential_alpha,
            BPR_ALPHA=bpr_alpha,
            BPR_BETA=bpr_beta
        )
        # グラフ上の移動時間を更新
        if weight == "travel_time":
            nx.set_edge_attributes(graph_with_time, {source_target: {'weight': after_bpr_time}})
        # 移動時間の更新
        current_travel_time_dict[source_target] = after_bpr_time

    # 時間によるシミュレート
    ctime=0

    evac_time = {}
    # 初期化済みのnow_dictを使う
    # シミュレーションの実行
    while(ctime <= max_time and fin_count < simulations):
        for i, current in enumerate(yet):
            if fin_evacuate[i]: # 避難完了した避難者はスキップ
                continue
            if wait_list[i] > 0: # 待ち時間がある場合
                wait_list[i] -= time_interval
                continue
            # 今のノードからどれだけ時間が経過したか
            current["from_index"] += time_interval
            # 時間が経過したら次のノードに移動
            if current["from_index"] >= current["need_time"]:
                source_target = current["node_to_node"]
                # 今のノードから出る人数をマイナス
                pass_count_dict[source_target] -= 1
                # 更新後移動時間を計算
                after_bpr_time = compute_travel_time(
                    n_di_dict[source_target], 
                    pass_count_dict[source_target], 
                    capacity_dict[source_target], 
                    model=model,
                    V_DEFAULT=speed_list[i],
                    V_MIN=v_min,
                    EXPONENTIAL_ALPHA=exponential_alpha,
                    BPR_ALPHA=bpr_alpha,
                    BPR_BETA=bpr_beta
                )
                # グラフ上の移動時間を更新
                if weight == "travel_time":
                    nx.set_edge_attributes(graph_with_time, {source_target: {'weight': after_bpr_time}})
                # 移動時間の更新
                current_travel_time_dict[source_target] = after_bpr_time
                if source_target[1] in (SINK_NODE_1, SINK_NODE_2): # 到達ノードがゴールであった場合
                    arrival_callback(i, simulations, fin_count, ctime)
                    # ゴール到達時間
                    evac_time[i] = ctime
                    # 避難完了した避難者を更新
                    fin_evacuate[i] = True
                    # 避難完了した避難者の数を更新
                    fin_count += 1
                    continue
                # 最適パスを選択
                if weight == "travel_time":
                    path = nx.dijkstra_path(
                        graph_with_time, 
                        source=source_target[1], # いま到着したノード
                        target=select_nearest_sink_node(graph_with_time, source_target[1]),
                        weight='weight'
                    )
                elif weight == "length":
                    path = paths[i][1]
                
                current["index"]+=1
                # 次のノードを選択
                if weight == "travel_time":
                    next_node = path[1]
                elif weight == "length":
                    next_node = path[current["index"]+1]
                # いま到着したノードから次のノード
                source_target = (source_target[1], next_node)
                current["node_to_node"] = source_target
                # その道を通る人数をカウント
                pass_count_dict[source_target] += 1
                # 自分が含まれていない移動時間
                current["need_time"] = current_travel_time_dict[source_target]
                # 更新後移動時間を計算
                after_bpr_time = compute_travel_time(
                    n_di_dict[source_target], 
                    pass_count_dict[source_target], 
                    capacity_dict[source_target], 
                    model=model,
                    V_DEFAULT=speed_list[i],
                    V_MIN=v_min,
                    EXPONENTIAL_ALPHA=exponential_alpha,
                    BPR_ALPHA=bpr_alpha,
                    BPR_BETA=bpr_beta
                )
                # グラフ上の移動時間を更新
                if weight == "travel_time":
                    nx.set_edge_attributes(graph_with_time, {source_target: {'weight': after_bpr_time}})
                # 移動時間の更新
                current_travel_time_dict[source_target] = after_bpr_time
                # 次のノードでの経過時間を0にリセット
                current["from_index"] = 0
        ctime += time_interval
    
    simulation_df = pd.DataFrame.from_dict(evac_time, orient='index', columns=['evac_time'])
    simulation_df['speed'] = simulation_df.index.map(speed_map)
    simulation_df['length'] = simulation_df.index.map(length_map)
    simulation_df['wait_time'] = simulation_df.index.map(wait_map)
    simulation_df['evac_time'] = simulation_df['evac_time'].astype(float)
    simulation_df = simulation_df.sort_values(by='evac_time')
    
    return simulation_df
