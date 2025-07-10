import hashlib

class DeterministicRandom:
    def __init__(self, seed):
        """ 環境に依存しない決定論的な乱数生成器 """
        self.seed = str(seed).encode()
        self.counter = 0

    def _hash(self):
        """ SHA-256 を用いた決定論的なハッシュベースの乱数 """
        self.counter += 1
        hash_value = hashlib.sha256(self.seed + str(self.counter).encode()).hexdigest()
        return int(hash_value[:8], 16) / 0xFFFFFFFF  # 32bit整数を浮動小数点に変換

    def randint(self, low, high):
        """ low 以上 high 以下の整数を返す """
        return low + int(self._hash() * (high - low + 1))

    def shuffle(self, array):
        """ Fisher-Yates シャッフル """
        arr = array[:]  # コピーを作成
        n = len(arr)
        for i in range(n - 1, 0, -1):
            j = self.randint(0, i)  # 0 以上 i 以下のランダムなインデックス
            arr[i], arr[j] = arr[j], arr[i]  # 要素をスワップ
        return arr

    def choice(self, array):
        """ 配列からランダムに 1 つの要素を選択 """
        index = self.randint(0, len(array) - 1)
        return array[index]

    def sample(self, array, k):
        """ 配列から k 個の要素をランダムに選択（重複なし） """
        if k > len(array):
            raise ValueError("k must be less than or equal to array length")
        
        arr = array[:]
        selected = []
        for _ in range(k):
            idx = self.randint(0, len(arr) - 1)
            selected.append(arr.pop(idx))  # 選択した要素をリストから削除
        return selected
