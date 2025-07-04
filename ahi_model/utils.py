def calcWaitInterval(interval: float, num_people: int, batch: int) -> list:
    """
    避難開始の待ち時間を計算する

    Parameters
    ----------
    interval : float
        避難開始の間隔
    num_people : int
        避難する人数
    batch : int
        一度に避難する人数

    Returns
    -------
    list
        避難開始の待ち時間のリスト
    """
    wait_times = []
    for i in range(num_people):
        wait_times.append(i // batch * interval)
    return wait_times
