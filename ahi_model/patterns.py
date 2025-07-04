from typing import Literal

Model = Literal['bpr', 'greenshields', 'exponential']
Weight = Literal['length', 'travel_time']
Order = Literal['near', 'far', 'random']

weight_map = {
    'length': '比較手法',
    'travel_time': '提案手法',
}

order_map = {
    'near': '近',
    'far': '遠',
    'random': 'ランダム',
}

TIME_INTERVAL = 0.1

DEFAULT_BPR_ALPHA = 1.1192
DEFAULT_BPR_BETA = 5.0365

def exponential(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speeds: list = [1.25],
        v_min: float = 0.03,
        exponential_alpha: float = 2.0,
        title: str = None,
        wait_times: list = [0.0],
) -> dict:
    # title = f'Exponential {weight.capitalize()} {order.capitalize()} {simulation} {human_speeds.__str__()} {v_min} {exponential_alpha}'
    if title is None:
        title = f'指数関数 {weight_map[weight]} {order_map[order]}'
    pattern = {
        'model': 'exponential',
        'human_speeds': human_speeds,
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'human_speed': human_speeds[0],
        'time_interval': TIME_INTERVAL,
        'v_min': v_min,
        'exponential_alpha': exponential_alpha,
        'bpr_alpha': DEFAULT_BPR_ALPHA,
        'bpr_beta': DEFAULT_BPR_BETA,
        'wait_times': wait_times,
    }
    return {
        'title': title,
        'pattern': pattern,
    }

def bpr(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speeds: list = [1.25],
        bpr_alpha: float = DEFAULT_BPR_ALPHA,
        bpr_beta: float = DEFAULT_BPR_BETA,
        title: str = None,
        wait_times: list = [0.0],
        v_min: float = 0.05
) -> dict:
    # title = f'BPR {weight.capitalize()} {order.capitalize()} {simulation} {human_speeds.__str__()} {bpr_alpha} {bpr_beta}'
    if title is None:
        title = f'BPR {weight_map[weight]} {order_map[order]}'
    pattern = {
        'human_speeds': human_speeds,
        'model': 'bpr',
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'time_interval': TIME_INTERVAL,
        'v_min': v_min,
        'exponential_alpha': 2.0,
        'bpr_alpha': bpr_alpha,
        'bpr_beta': bpr_beta,
        'wait_times': wait_times,
    }
    return {
        'title': title,
        'pattern': pattern,
    }

def greenshields(
        weight: Weight = 'length',
        order: Order = 'near',
        simulation: int = 1000,
        human_speeds: list = [1.25],
        v_min: float = 0.03,
        title: str = None,
        wait_times: list = [0.0],
) -> dict:
    # title = f'Greenshields {weight.capitalize()} {order.capitalize()} {simulation} {human_speeds.__str__()} {v_min}'
    if title is None:
        title = f'グリーンシールズ {weight_map[weight]} {order_map[order]}'
    pattern = {
        'model': 'greenshields',
        'weight': weight,
        'order': order,
        'simulation': simulation,
        'human_speeds': human_speeds,
        'time_interval': TIME_INTERVAL,
        'v_min': v_min,
        'exponential_alpha': 2.0,
        'bpr_alpha': DEFAULT_BPR_ALPHA,
        'bpr_beta': DEFAULT_BPR_BETA,
        'wait_times': wait_times,
    }
    return {
        'title': title,
        'pattern': pattern,
    }
