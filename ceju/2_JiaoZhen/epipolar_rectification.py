import numpy as np
import cv2
import os


def undistort_and_remap(image, camera_matrix, distortion, R, P, size):
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, distortion, R, P, size, cv2.CV_32FC1
    )
    rectified_image = cv2.remap(image, map1, map2, cv2.INTER_CUBIC)
    return rectified_image


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20, 3), dtype=np.uint8)
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg

    for i in range(int(HEIGHT / 32)):
        imgcat[i * 32, :, :] = 255

    return imgcat


# -----------------------------
# 读取左右图（你只需要修改这里）
# -----------------------------
left_image_path = "left_19.jpg"
right_image_path = "right_19.jpg"

left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

if left_image is None or right_image is None:
    raise ValueError("无法读取左右图片，请检查路径是否正确！")

HEIGHT = left_image.shape[0]
WIDTH = left_image.shape[1]

# 保存目录
save_folder = "rectified_images"
os.makedirs(save_folder, exist_ok=True)

# 拼接原始图
imgcat_source = cat2images(left_image, right_image)
cv2.imwrite(os.path.join(save_folder, 'imgcat_source.jpg'), imgcat_source)


# ============================
#     相机标定参数（与你一致）
# ============================
camera_matrix0 = np.array([
    [1.057389594483230e+03, 1.698879225510185, 5.800114785214288e+02],
    [0, 1.058384060607384e+03, 5.084522757265003e+02],
    [0, 0, 1]
])

distortion0 = np.array([[0.048033739382123, 0.311433858924166,
                          0.001205307356943, 0.001270572815158,
                          -0.385926882800597]])

camera_matrix1 = np.array([
    [1.065085617396094e+03, 1.384118782509674, 5.809912806789391e+02],
    [0, 1.065864639143073e+03, 4.958062841067978e+02],
    [0, 0, 1]
])

distortion1 = np.array([[-0.049056548243087, 0.345747823496975,
                          0.001058038492965, 0.001167808554822,
                          -0.461392018886485]])

R = np.array([
    [1, 0.001101971133142, 8.064727977011493e-04],
    [-0.001098406406917, 1, -0.004407319052475],
    [-8.113212168754618e-04, 0.004406429108336, 1]
])

T = np.array([[-61.340492524588896],
              [-0.114434490871765],
              [1.758889832299615]])


# --------------------------
