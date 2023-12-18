import datetime
import os

# 画像ファイルを開く
# path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\*.jpg'
path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131338724_V.jpg"

s = os.path.basename(path)[:-6]
time = datetime.datetime.strptime(s, '%Y%m%d_%H%M%S%f')
print(time)