import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_fontja
from simulate import simulate
from patterns import *
from utils import *
import time

# 避難経路長、避難速度、待ち時間をどの単位で平滑化するか(平均取るか)
CHUNK_SIZE = 10

# 避難人数
NUM_OF_PEOPLE = 50000

# 必要ファイル
NEEDS = [
    'simulations',
    # 'time',
    # 'speed',
    # 'length',
    # 'wait_time'
]

# ファイル名のプレフィックス(先頭につける文字列)
PREFIX = f"{NUM_OF_PEOPLE}_"
# ファイル名のサフィックス(末尾につける文字列)
SUFFIX = '_order_compare'

### WAIT PATTERNS ###
# 何秒おきに人が出発するか
WAIT_INTERVAL = 60 # 3分おき
# 何人が同時に出発するか
BATCH_SIZE = int(NUM_OF_PEOPLE * 0.05) # 25%

# wait_times = calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)


patterns = [
    # bpr('travel_time', 'near', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 近', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),
    # bpr('travel_time', 'far', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 遠', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),
    # bpr('travel_time', 'random', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 ランダム', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),

    # bpr('length', 'near', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法 近', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),
    # bpr('length', 'far', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法 遠', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),
    # bpr('length', 'random', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法 ランダム', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),

    # bpr('travel_time', 'near', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),
    # bpr('length', 'near', NUM_OF_PEOPLE, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, BATCH_SIZE)),

    # bpr('travel_time', 'near', 1000, [1.25], 1.919, 6.9373, '1000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 2000, [1.25], 1.919, 6.9373, '2000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 3000, [1.25], 1.919, 6.9373, '3000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 4000, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 5000, [1.25], 1.919, 6.9373, '5000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 6000, [1.25], 1.919, 6.9373, '6000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 1600, [1.25], 1.919, 6.9373, '1600', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'near', 1700, [1.25], 1.919, 6.9373, '1700', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),


    bpr('travel_time', 'far', 1000, [1.25], 1.919, 6.9373, '1000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 2000, [1.25], 1.919, 6.9373, '2000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 3000, [1.25], 1.919, 6.9373, '3000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4000, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 5000, [1.25], 1.919, 6.9373, '5000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 5000, [1.25], 1.919, 6.9373, '提案手法 遠', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE),10e-54),
    # bpr('travel_time', 'near', 50000, [1.25], 1.1192, 5.0365, '提案手法 近', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE),10e-54),
    # bpr('length', 'near', 5000, [1.25], 1.919, 6.9373, '比較手法', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE),0.04),
    # bpr('travel_time', 'far', 4100, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4200, [1.25], 1.919, 6.9373, '4200', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4300, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4400, [1.25], 1.919, 6.9373, '4400', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4500, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4600, [1.25], 1.919, 6.9373, '4600', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4700, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4800, [1.25], 1.919, 6.9373, '4800', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 4900, [1.25], 1.919, 6.9373, '4000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
    # bpr('travel_time', 'far', 5000, [1.25], 1.919, 6.9373, '5000', calcWaitInterval(WAIT_INTERVAL, NUM_OF_PEOPLE, NUM_OF_PEOPLE)),
]

time_df = pd.DataFrame()
speed_df = pd.DataFrame()
length_df = pd.DataFrame()
wait_time_df = pd.DataFrame()
res_df = pd.DataFrame()
max_map = {}
for i, pt in enumerate(patterns):
    print(f"{pt['title']}")
    p = pt['pattern']
    start_time = time.time()
    res = simulate(
        csv_file='suzu_edges_modified.csv',
        human_speeds=p['human_speeds'],
        model=p['model'],
        weight=p['weight'],
        order=p['order'],
        simulations=p['simulation'],
        default_human_speed=p['human_speeds'][0],
        wait_times=p['wait_times'],
        time_interval=p['time_interval'],
        v_min=p['v_min'],
        exponential_alpha=p['exponential_alpha'],
        bpr_alpha=p['bpr_alpha'],
        bpr_beta=p['bpr_beta']
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time}")
    time_df[f'{pt["title"]}'] = [elapsed_time]

    ev_speed = res['speed']
    ev_speed.index = np.arange(1, len(ev_speed)+1)
    speed_df[f'{pt["title"]}'] = ev_speed

    ev_length = res['length']
    ev_length.index = np.arange(1, len(ev_length)+1)
    length_df[f'{pt["title"]}'] = ev_length

    ev_wait_time = res['wait_time']
    ev_wait_time.index = np.arange(1, len(ev_wait_time)+1)
    wait_time_df[f'{pt["title"]}'] = ev_wait_time
    
    ev_time = res['evac_time']
    ev_time.index = np.arange(1, len(ev_time)+1)
    res_df[f'{pt["title"]}'] = ev_time

    max_evac_time = res['evac_time'].max()
    max_map[pt['pattern']['simulation']] = max_evac_time

# max_df = pd.DataFrame.from_dict(max_map, orient='index')
# max_df.to_csv(f"outputs/{PREFIX}max{SUFFIX}.csv", index=False, encoding='utf-8-sig')

# plt.figure(figsize=(8, 5))
# plt.plot(max_df.index, max_df[0], marker='o', markersize=1)

# # 軸ラベルとタイトル
# plt.xlabel('避難人数')
# plt.ylabel('避難時間（秒）')
# plt.title("最大避難時間")
# plt.grid(True)

# # グラフを表示
# plt.savefig(f"plot/{PREFIX}max{SUFFIX}.png")

if 'simulations' in NEEDS:
    res_df.to_csv(f"outputs/{PREFIX}simulations{SUFFIX}.csv", index=False, encoding='utf-8-sig')
if 'time' in NEEDS:
    time_df.to_csv(f"outputs/{PREFIX}time{SUFFIX}.csv", index=False, encoding='utf-8-sig')
if 'speed' in NEEDS:
    speed_df.to_csv(f"outputs/{PREFIX}speed{SUFFIX}.csv", index=False, encoding='utf-8-sig')
if 'length' in NEEDS:
    length_df.to_csv(f"outputs/{PREFIX}length{SUFFIX}.csv", index=False, encoding='utf-8-sig')
if 'wait_time' in NEEDS:
    wait_time_df.to_csv(f"outputs/{PREFIX}wait_time{SUFFIX}.csv", index=False, encoding='utf-8-sig')

if 'simulations' in NEEDS:
    res_df = pd.read_csv(f"outputs/{PREFIX}simulations{SUFFIX}.csv")

    plt.figure(figsize=(8, 5))
    for column in res_df.columns:
        plt.plot(res_df.index, res_df[column], marker='o', label=column, markersize=1)

    # 軸ラベルとタイトル
    plt.xlabel('避難人数')
    plt.ylabel('避難時間（秒）')
    plt.title("シミュレーション結果")
    plt.legend()
    plt.grid(True)

    # グラフを表示
    plt.savefig(f"plot/{PREFIX}simulations{SUFFIX}.png")

if 'length' in NEEDS:
    length_df = pd.read_csv(f"outputs/{PREFIX}length{SUFFIX}.csv")

    # indexが到着順位となっている
    length_df["到着順位"] = length_df.index

    # 順位100区切りの平均を取得
    length_df["到着順位"] = length_df["到着順位"].apply(lambda x: x//CHUNK_SIZE * CHUNK_SIZE)
    length_df = length_df.groupby("到着順位").mean()

    plt.figure(figsize=(8, 5))
    for column in length_df.columns:
        plt.plot(length_df.index, length_df[column], marker='o', label=column, markersize=1)

    # 軸ラベルとタイトル
    plt.xlabel('到着順位')
    plt.ylabel('避難経路長（m）')
    plt.title("避難経路長")
    plt.legend()
    plt.grid(True)

    # グラフを表示
    plt.savefig(f"plot/{PREFIX}length{SUFFIX}.png")

if 'speed' in NEEDS:
    speed_df = pd.read_csv(f"outputs/{PREFIX}speed{SUFFIX}.csv")

    # indexが到着順位となっている
    speed_df["到着順位"] = speed_df.index

    # 順位100区切りの平均を取得
    speed_df["到着順位"] = speed_df["到着順位"].apply(lambda x: x//CHUNK_SIZE * CHUNK_SIZE)
    speed_df = speed_df.groupby("到着順位").mean()

    plt.figure(figsize=(8, 5))
    for column in speed_df.columns:
        plt.plot(speed_df.index, speed_df[column], marker='o', label=column, markersize=1)

    # 軸ラベルとタイトル
    plt.xlabel('到着順位')
    plt.ylabel('避難速度（m/s）')
    plt.title("避難速度")
    plt.legend()
    plt.grid(True)

    # グラフを表示
    plt.savefig(f"plot/{PREFIX}speed{SUFFIX}.png")

if 'wait_time' in NEEDS:
    wait_time_df = pd.read_csv(f"outputs/{PREFIX}wait_time{SUFFIX}.csv")

    # indexが到着順位となっている
    wait_time_df["到着順位"] = wait_time_df.index

    # 順位100区切りの平均を取得
    wait_time_df["到着順位"] = wait_time_df["到着順位"].apply(lambda x: x//CHUNK_SIZE * CHUNK_SIZE)
    wait_time_df = wait_time_df.groupby("到着順位").mean()

    plt.figure(figsize=(8, 5))
    for column in wait_time_df.columns:
        plt.plot(wait_time_df.index, wait_time_df[column], marker='o', label=column, markersize=1)

    # 軸ラベルとタイトル
    plt.xlabel('到着順位')
    plt.ylabel('待ち時間（秒）')
    plt.title("待ち時間")
    plt.legend()
    plt.grid(True)

    # グラフを表示
    plt.savefig(f"plot/{PREFIX}wait_time{SUFFIX}.png")
