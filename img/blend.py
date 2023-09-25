import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt


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

# エンターおすと画像重ね合わせ
def onEnter():
    # P:変換元の座標行列([[x,y,1],[x,y,1],...]
    # x_dash:変換先のx座標ベクトル
    # y_dash:変換先のy座標ベクトル
    affine_matrix = np.array([[ 8.63268768e-01, -1.52225002e-02, 7.40494984e+01],
    [ 2.65879935e-02, 8.55377241e-01, 3.98972349e+01],
    [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    #アフィン変換後の画像求める
    affined_image = affine(image2, affine_matrix, image1.shape)
    x = np.arange(0, affined_image.shape[1], 1)
    y = np.arange(0, affined_image.shape[0], 1)
    X_affined, Y_affined = np.meshgrid(x, y)
    ax3.pcolormesh(X_affined, Y_affined, affined_image, cmap='gray', shading='auto', alpha=0.3)
    # fig.canvas.draw_idle()


# 画像読み込み
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.jpg"
visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358202_V.jpg"

image1 = np.array(Image.open(a_thermal_path).convert('L'))
image2 = np.array(Image.open(visible_path).convert('L'))
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
# ax1 = fig.add_subplot(221)
# mesh1 = ax1.pcolormesh(X1, Y1, image1, shading='auto', cmap='gray')
# ax1.invert_yaxis()
# ax2 = fig.add_subplot(223)
# mesh2 = ax2.pcolormesh(X2, Y2, image2, shading='auto', cmap='gray')
# ax2.invert_yaxis()
ax3 = fig.add_subplot(111)
mesh3 = ax3.pcolormesh(X1, Y1, image1, shading='auto', cmap='gray', alpha=1)
ax3.invert_yaxis()

onEnter()
plt.show()
