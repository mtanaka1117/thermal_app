import numpy as np


with open('.data.dat', 'rb') as f:
    binary_data = f.read()

# バイナリデータをNumPy配列に変換
# 元の配列の形状とデータ型を知っている場合
data_restored = np.frombuffer(binary_data, dtype=np.uint8).reshape((2, 3))

print(data_restored.dtype)

# 例として、NumPy配列を作成
data_restored = data_restored + 5

# NumPy配列をバイナリデータに変換
binary_data = data_restored.tobytes()

# バイナリデータをファイルに保存
with open('.data2.dat', 'wb') as f:
    f.write(binary_data)


with open('.data2.dat', 'rb') as f:
    binary_data = f.read()

# バイナリデータをNumPy配列に変換
# 元の配列の形状とデータ型を知っている場合
data_restored = np.frombuffer(binary_data, dtype=np.uint8).reshape((2, 3))

print(data_restored.dtype)