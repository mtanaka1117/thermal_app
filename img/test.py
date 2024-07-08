import numpy as np

# NumPy配列を作成
data = np.array([5, 10, 15, 20])

# 閾値を設定
threshold = 12

# 閾値以下の値をフィルタリング
filtered_data = data[data <= threshold]

print(data <= threshold)
print(filtered_data)