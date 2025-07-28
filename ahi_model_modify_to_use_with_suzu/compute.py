import math

V_DEFAULT = 1.25  # 自由走行速度
V_MIN = 0.05 # 最小速度

EXPONENTIAL_ALPHA = 2.0

# BPR_ALPHA = 1.1192
# BPR_BETA = 5.0365


def compute_travel_time(L, N, C, model='bpr', V_DEFAULT=V_DEFAULT, V_MIN=V_MIN, EXPONENTIAL_ALPHA=EXPONENTIAL_ALPHA, BPR_ALPHA=1.1192, BPR_BETA=5.0365) -> float:
    if model == 'bpr':
        return min(travel_time_bpr(L/V_DEFAULT, N, C, BPR_ALPHA, BPR_BETA), L/V_MIN)
    elif model == 'greenshields':
        return travel_time_greenshields(L, N, C, V_DEFAULT, V_MIN)
    elif model == 'exponential':
        return travel_time_exponential(L, N, C, V_DEFAULT, V_MIN, EXPONENTIAL_ALPHA)
    else:
        raise ValueError('Invalid model')


def travel_time_greenshields(L, N, C, V_DEFAULT=V_DEFAULT, V_MIN=V_MIN) -> float:
    """
    グリーンシールズの交通流モデルに基づく移動時間の計算
    L: 道路の長さ
    N: 道路上の車両数
    C: 道路の容量

    Returns 移動時間
    """
    V_prime = max(V_DEFAULT * (1 - (N / C) ** 2), V_MIN)
    return L / V_prime

def travel_time_exponential(L, N, C, V_DEFAULT=V_DEFAULT, V_MIN=V_MIN, EXPONENTIAL_ALPHA=EXPONENTIAL_ALPHA) -> float:
    V_prime = max(V_DEFAULT * math.exp(-EXPONENTIAL_ALPHA * (N / C)), V_MIN)
    return L / V_prime

def travel_time_bpr(T0, N, C, BPR_ALPHA=1.1192, BPR_BETA=5.0365) -> float:
    """
    BPRモデルに基づく移動時間の計算（安全対策あり）
    T0: 渋滞していないときの移動時間
    N: 道路上の車両数
    C: 道路の容量
    """
    if C <= 0:
        return float("inf")  # 容量为0时，表示完全堵塞

    ratio = N / C
    if ratio < 0:
        ratio = 0  # 防止负数幂引发复数

    try:
        t = T0 * (1 + BPR_ALPHA * (ratio ** BPR_BETA))
        return t
    except Exception:
        return float("inf")