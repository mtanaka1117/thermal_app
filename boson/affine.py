# python C:\opencv_build\python_loader\setup.py install --user
# 参考　https://qiita.com/konnitiha/items/bd8bc5823247fed8368c
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt


# 参照画像の範囲超えたやつは配列の最後を参照するようにする関数
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


# 特徴点からアフィン行列求める関数
def registration(P, x_dash, y_dash):
    w1 = np.linalg.inv(P.T @ P) @ P.T @ x_dash
    w2 = np.linalg.inv(P.T @ P) @ P.T @ y_dash
    affine_matrix = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    affine_matrix[0, :] = w1
    affine_matrix[1, :] = w2
    print(affine_matrix)
    print("逆行列", np.linalg.inv(affine_matrix))
    return affine_matrix


# クリックした特徴点保存する配列
future_points1 = np.array([[1, 1]])
future_points2 = np.array([[1, 1]])
count_fp1 = 0
count_fp2 = 0


# クリックで特徴点決める
def onclick(event):
    global future_points1
    global future_points2
    global count_fp1
    global count_fp2

    click_axes = event.inaxes
    x = math.floor(event.xdata)
    y = math.floor(event.ydata)
    if click_axes == ax1:
        if count_fp1 == 0:
            future_points1[0, :] = (x, y)
            count_fp1 = 1
        else:
            future_points1 = np.vstack([future_points1, np.array([x, y])])
            count_fp1 += count_fp1
        print(future_points1)
    if click_axes == ax2:
        if count_fp2 == 0:
            future_points2[0, :] = (x, y)
            count_fp2 = 1
        else:
            future_points2 = np.vstack([future_points2, np.array([x, y])])
            count_fp2 += count_fp2
        print(future_points2)
    click_axes.scatter(x, y)
    fig.canvas.draw_idle()


# エンターおすと画像重ね合わせ
def onEnter(event):
    if event.key == 'enter' and future_points1.size == future_points2.size and future_points1.size >= 3:
        # P:変換元の座標行列([[x,y,1],[x,y,1],...]
        # x_dash:変換先のx座標ベクトル
        # y_dash:変換先のy座標ベクトル
        vec_one = np.ones((future_points2.shape[0], 1))
        P = np.hstack([future_points2, vec_one])
        x_dash = future_points1[:, 0]
        y_dash = future_points1[:, 1]
        affine_matrix = registration(P, x_dash, y_dash)

        #アフィン変換後の画像求める
        affined_image = affine(image2, affine_matrix, image1.shape)
        x = np.arange(0, affined_image.shape[1], 1)
        y = np.arange(0, affined_image.shape[0], 1)
        X_affined, Y_affined = np.meshgrid(x, y)
        ax3.pcolormesh(X_affined, Y_affined, affined_image, cmap='gray', shading='auto', alpha=0.2)
        fig.canvas.draw_idle()


# 画像読み込み
# thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.jpg"
# visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358202_V.jpg"

# thermal_path = r"C:\Users\tnkmo\Downloads\20240112_items2-selected\yolo1\20240112_1418\20240112_141859956_T.jpg"
# visible_path = r"C:\Users\tnkmo\Downloads\20240112_items2-selected\yolo1\20240112_1418\20240112_141859978_V.jpg"

thermal_path = r"C:\Users\tnkmo\OneDrive\デスクトップ\thermal_app\boson\data\0608\table\20240608_111002791_T.jpg"
visible_path = r"C:\Users\tnkmo\OneDrive\デスクトップ\thermal_app\boson\data\0608\table\20240608_111002791_V.jpg"


image1 = np.array(Image.open(thermal_path).convert('L'))
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
ax1 = fig.add_subplot(221)
mesh1 = ax1.pcolormesh(X1, Y1, image1, shading='auto', cmap='gray')
ax1.invert_yaxis()
ax2 = fig.add_subplot(223)
mesh2 = ax2.pcolormesh(X2, Y2, image2, shading='auto', cmap='gray')
ax2.invert_yaxis()
ax3 = fig.add_subplot(222)
mesh3 = ax3.pcolormesh(X1, Y1, image1, shading='auto', cmap='gray', alpha=0.2)
ax3.invert_yaxis()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', onEnter)
plt.show()