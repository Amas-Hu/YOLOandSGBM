import numpy as np

def compute_q_matrix(left_cam_matrix, right_cam_matrix, T):
    """
    计算双目相机的重投影矩阵Q
    
    参数：
        left_cam_matrix: 左相机内参矩阵 (3x3)，格式为[[fx, 0, cx],
                                                    [0, fy, cy],
                                                    [0, 0, 1]]
        right_cam_matrix: 右相机内参矩阵 (3x3)，格式同上
        T: 右相机相对于左相机的平移向量 (3x1)，通常为[x, y, z]，x分量的绝对值为基线B
    
    返回：
        Q: 4x4重投影矩阵
    """
    # 提取左相机内参
    fx = left_cam_matrix[0, 0]  # x方向焦距
    cx = left_cam_matrix[0, 2]  # 左相机主点x坐标
    cy = left_cam_matrix[1, 2]  # 左相机主点y坐标
    
    # 提取右相机主点x坐标
    cx_prime = right_cam_matrix[0, 2]
    
    # 计算基线B（平移向量T的x分量的绝对值）
    B = abs(T[0])
    if B < 1e-6:
        raise ValueError("基线距离B过小，可能输入的平移向量有误")
    
    # 构造Q矩阵
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1/B, (cx - cx_prime)/B]
    ], dtype=np.float64)
    
    return Q

# -------------------------- 示例：使用实际参数计算Q矩阵 --------------------------
if __name__ == "__main__":
    # 1. 输入相机参数（替换为你的标定结果）
    # 左相机内参矩阵 (fx, cx, cy 从内参中提取)
    left_cam_matrix = np.array([
        [   1.489033246995980e+03,                         0,          9.786633681175864e+02],
        [                       0,     1.487981201679089e+03,          5.502148568929165e+02],
        [                       0,                         0,                              1]
    ])
    
    # 右相机内参矩阵
    right_cam_matrix = np.array([
        [   1.493353756536974e+03,                          0,          9.510105842885547e+02],
        [                       0,      1.494819893020159e+03,          5.638273549540610e+02],
        [                       0,                          0,                              1]
    ])
    
    # 右相机相对于左相机的平移向量T（Tx, Ty, Tz）
    T = np.array([  -1.201637642763763e+02,   0.136204017162100,    0.202319355852273])  # 单位：毫米
    
    # 2. 计算Q矩阵
    Q = compute_q_matrix(left_cam_matrix, right_cam_matrix, T)
    
    # 3. 输出结果
    print("计算得到的Q矩阵（重投影矩阵）：")
    print(Q)
    
    # 4. （可选）保存Q矩阵到文件
    np.save("q_matrix.npy", Q)
    print("\nQ矩阵已保存为 q_matrix.npy")