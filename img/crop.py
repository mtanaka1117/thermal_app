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
feature_points1 = np.array([[1, 1]])
count_fp1 = 0


# クリックで特徴点決める
def onclick(event):
    global feature_points1
    global count_fp1

    click_axes = event.inaxes
    x = math.floor(event.xdata)
    y = math.floor(event.ydata)
    if click_axes == ax1:
        if count_fp1 == 0:
            feature_points1[0, :] = (x, y)
            count_fp1 = 1
        else:
            feature_points1 = np.vstack([feature_points1, np.array([x, y])])
            count_fp1 += count_fp1
        print(feature_points1)
    click_axes.scatter(x, y)
    fig.canvas.draw_idle()




img_path = '20241223_203142700_V.jpg'
img = np.array(Image.open(img_path).convert('L'))

bg_color = 256
img = np.hstack([img, bg_color * np.ones((img.shape[0], 1), int)])
img = np.vstack([img, bg_color * np.ones((1, img.shape[1]), int)])

x_img = np.arange(0, img.shape[1], 1)
y_img = np.arange(0, img.shape[0], 1)

X, Y = np.meshgrid(x_img, y_img)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot()
mesh1 = ax1.pcolormesh(X, Y, img, shading='auto', cmap='gray')
ax1.invert_yaxis()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()



with Image.open(img_path) as img:
    left_x, left_y = feature_points1[0]
    right_x, right_y = feature_points1[1]
    cropped_img = img.crop((left_x, left_y, right_x, right_y))
    cropped_img.save('cropped_image.jpg')
    print('image saved')
