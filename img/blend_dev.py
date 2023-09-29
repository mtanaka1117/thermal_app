import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


# 参照画像の範囲超えた分は配列の最後を参照するようにする関数
def clip_xy(ref_xy, img_shape):
    # x座標について置換
    ref_x = np.where((0 <= ref_xy[:, 0]) & (ref_xy[:, 0] < img_shape[1]), ref_xy[:, 0], -1)
    # y座標について置換
    ref_y = np.where((0 <= ref_xy[:, 1]) & (ref_xy[:, 1] < img_shape[0]), ref_xy[:, 1], -1)
    # 結合して返す
    return np.vstack([ref_x, ref_y]).T

# アフィン変換
def affine(data, affine, draw_area_size):
    # data:アフィン変換させる画像データ
    # affine:アフィン行列
    #:draw_area_size:dataのshapeと同じかそれ以上がいいかも

    # アフィン行列の逆行列
    inv_affine = np.linalg.inv(affine)

    x = np.arange(0, draw_area_size[1], 1)
    y = np.arange(0, draw_area_size[0], 1)
    X, Y = np.meshgrid(x, y)

    XY = np.dstack([X, Y, np.ones_like(X)])
    xy = XY.reshape(-1, 3).T

    # 参照座標の計算
    ref_xy = inv_affine @ xy
    ref_xy = ref_xy.T

    # 参照座標の周りの座標
    liner_xy = {}
    liner_xy['downleft'] = ref_xy[:, :2].astype(int)
    liner_xy['upleft'] = liner_xy['downleft'] + [1, 0]
    liner_xy['downright'] = liner_xy['downleft'] + [0, 1]
    liner_xy['upright'] = liner_xy['downleft'] + [1, 1]

    # 線形補間での重み計算
    liner_diff = ref_xy[:, :2] - liner_xy['downleft']

    liner_weight = {}
    liner_weight['downleft'] = (1 - liner_diff[:, 0]) * (1 - liner_diff[:, 1])
    liner_weight['upleft'] = (1 - liner_diff[:, 0]) * liner_diff[:, 1]
    liner_weight['downright'] = liner_diff[:, 0] * (1 - liner_diff[:, 1])
    liner_weight['upright'] = liner_diff[:, 0] * liner_diff[:, 1]

    # 重み掛けて足す
    liner_with_weight = {}
    for direction in liner_weight.keys():
        l_xy = liner_xy[direction]
        l_xy = clip_xy(l_xy, data.shape)
        l_xy = np.dstack([l_xy[:, 0].reshape(draw_area_size), l_xy[:, 1].reshape(draw_area_size)])
        l_weight = liner_weight[direction].reshape(draw_area_size)
        liner_with_weight[direction] = data[l_xy[:, :, 1], l_xy[:, :, 0]] * l_weight

    data_linear = sum(liner_with_weight.values())
    return data_linear

# 画像重ね合わせ
def onEnter():
    # affine_matrix = np.array([[ 8.63268768e-01, -1.52225002e-02, 7.40494984e+01],
    # [ 2.65879935e-02, 8.55377241e-01, 3.98972349e+01],
    # [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                            [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01],
                            [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    #アフィン変換後の画像求める
    affined_image = affine(image2, affine_matrix, image1.shape)
    x = np.arange(0, affined_image.shape[1], 1)
    y = np.arange(0, affined_image.shape[0], 1)
    X_affined, Y_affined = np.meshgrid(x, y)
    ax.pcolormesh(X_affined, Y_affined, affined_image, cmap='gray', shading='auto', alpha=0.8)

# 閾値で二値化
def binalization(data, threshold):
    img = data.copy()
    img[data.astype(np.uint8) < threshold] = 0
    img[data.astype(np.uint8) >= threshold] = 255
    return img

# 各画素16bit値、バイナリデータから画像を取得
# np.frombeffer()はbufferをarrayとして解釈 https://deepage.net/features/numpy-frombuffer.html
def thermal_image(path, path2):
    with open(path, "rb") as f:
        img_binary = f.read()
    data = np.frombuffer(img_binary, dtype = np.uint16).reshape([512, 640])/10 - 273.15

    with open(path2, "rb") as f:
        img_binary2 = f.read()
    data2 = np.frombuffer(img_binary2, dtype = np.uint16).reshape([512, 640])/10 - 273.15

    img_diff = cv2.absdiff(data, data2)
    _, img_th = cv2.threshold(img_diff,20,255,cv2.THRESH_BINARY)
    # image = Image.fromarray(img_th.astype(np.uint8))
    return img_th

# 画像読み込み
b_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354592_T.dat"
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.dat"
# a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.jpg"
visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358202_V.jpg"

image2 = thermal_image(b_thermal_path, a_thermal_path)
# image1 = np.array(Image.open(a_thermal_path).convert('L'))
image1 = np.array(Image.open(visible_path).convert('L'))

# 画像の最後にbg_colorの色追加
bg_color = 256
image2 = np.hstack([image2, bg_color * np.ones((image2.shape[0], 1), int)])
image2 = np.vstack([image2, bg_color * np.ones((1, image2.shape[1]), int)])

x_image1 = np.arange(0, image1.shape[1], 1)
y_image1 = np.arange(0, image1.shape[0], 1)
X1, Y1 = np.meshgrid(x_image1, y_image1)

x_image2 = np.arange(0, image2.shape[1], 1)
y_image2 = np.arange(0, image2.shape[0], 1)
X2, Y2 = np.meshgrid(x_image2, y_image2)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
mesh = ax.pcolormesh(X1, Y1, image1, shading='auto', cmap='gray', alpha=1)
ax.invert_yaxis()

onEnter()
plt.show()
