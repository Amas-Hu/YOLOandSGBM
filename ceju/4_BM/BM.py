# BM (Block Matching) 双目测距算法 v2
# 基于camera_configs.py和BM.py实现，支持1920x1080分辨率图像

import cv2
import numpy as np
import math
import os
import datetime as dt

# 相机参数配置
class CameraConfigs:
    def __init__(self):
        # 左相机内参矩阵
        # 构成形式: [fx_left, skew_left, cx_left]
        #         [0,       fy_left,  cy_left]
        #         [0,        0,        1    ]
        # 其中: fx_left=986.4572391 (x轴焦距)
        #      skew_left=1.673607456 (像素偏斜系数)
        #      cx_left=651.0717611 (x轴主点坐标)
        #      fy_left=1001.238398 (y轴焦距)
        #      cy_left=535.8195077 (y轴主点坐标)
        self.left_camera_matrix = np.array([[1.481512980952683e+03,                     0, 9.741930028279249e+02],
                                            [                    0, 1.480735907711620e+03, 5.435121362485065e+02],
                                            [                    0,                     0,                      1]])
        # 左相机畸变系数 [k1, k2, p1, p2, k3]
        # k1, k2, k3: 径向畸变系数
        # p1, p2: 切向畸变系数
        self.left_distortion = np.array([[0.302369610254450, -0.832843922384300, 0.001560565758676, -0.005541012945384]])
        
        # 右相机内参矩阵
        # 构成形式: [fx_right, skew_right, cx_right]
        #         [0,        fy_right,  cy_right]
        #         [0,         0,         1     ]
        # 其中: fx_right=998.5848065 (x轴焦距)
        #      skew_right=7.37746018 (像素偏斜系数)
        #      cx_right=667.3698587 (x轴主点坐标)
        #      fy_right=1006.305891 (y轴焦距)
        #      cy_right=528.9731771 (y轴主点坐标)
        self.right_camera_matrix = np.array([[1.482296723453751e+03,                     0, 9.660834185159497e+02],
                                            [                     0, 1.482629468412100e+03, 5.492517774177843e+02],
                                            [                     0,                     0,                     1]])
        # 右相机畸变系数 [k1, k2, p1, p2, k3]
        self.right_distortion = np.array([[0.294255745818360, 0.033358045248374, -0.001227604040424, 0.002781368162044]])
        
        # 旋转矩阵 (3x3)
        # 构成形式: [R11, R12, R13]
        #         [R21, R22, R23]
        #         [R31, R32, R33]
        # 表示从右相机坐标系到左相机坐标系的旋转变换
        self.R = np.array([[    0.999907272711480,  -9.336217474162630e-04,  -0.013585813524483],
                        [   9.689518153560432e-04,       0.999996165720932,   0.002594161870988],
                        [       0.013583339466744,      -0.002607085320069,   0.999904343422442]])
        
        # 平移矩阵 (3x1)
        # 构成形式: [Tx, Ty, Tz]
        # 表示从右相机坐标系到左相机坐标系的平移变换，单位为mm
        # Tx=-117.3364039 (左右方向平移)
        # Ty=0.277054571 (上下方向平移)
        # Tz=-3.7672413 (前后方向平移)
        self.T = np.array([ -1.202021696951524e+02, -0.010096027651827, -0.512882222667619])
        
        # 图像尺寸 - OpenCV中使用(width, height)格式
        # 因此(1920, 1080)表示宽度为1920像素，高度为1080像素
        self.size = (1920, 1080)
        
        # 立体校正
        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = \
            cv2.stereoRectify(self.left_camera_matrix, self.left_distortion,
                            self.right_camera_matrix, self.right_distortion, 
                            self.size, self.R, self.T)
        
        # 校正查找映射表
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion, self.R1, self.P1, self.size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion, self.R2, self.P2, self.size, cv2.CV_16SC2)
        
        # 焦距 (mm) - 使用内参矩阵中的值计算
        self.focal_length = (self.left_camera_matrix[0, 0] + self.right_camera_matrix[0, 0]) / 2
        
        # 基线距离 (mm) - 使用平移矩阵中的值
        self.baseline = abs(self.T[0])

# BM立体匹配类
class BlockMatcher:
    def __init__(self, camera_configs):
        self.camera_configs = camera_configs
        self.stereo = self._init_stereo_bm()
    
    def _init_stereo_bm(self):
        """初始化StereoBM对象并设置参数"""
        # 计算合适的视差数量，针对1920x1080图像优化 - 增加视差数量以提高远距离检测能力
        numberOfDisparities = 256  # 设置固定值以获得更好的远距离检测
        print(f"视差数量: {numberOfDisparities}")
        
        # 创建StereoBM对象
        stereo = cv2.StereoBM_create(numDisparities=numberOfDisparities, blockSize=15)  # 增加块大小以提高稳定性
        
        # 设置ROI以限制计算区域，提高精度
        if self.camera_configs.validPixROI1 is not None:
            stereo.setROI1(self.camera_configs.validPixROI1)
        if self.camera_configs.validPixROI2 is not None:
            stereo.setROI2(self.camera_configs.validPixROI2)
        
        # 设置BM算法参数 - 优化以提高视差质量和前后景分离效果
        stereo.setPreFilterType(1)  # 使用基本预过滤
        stereo.setPreFilterCap(31)        # 降低预过滤帽子以保留更多纹理信息
        stereo.setBlockSize(11)           # 增加块大小以提高稳定性和远距离检测
        stereo.setMinDisparity(4)         # 设置非零最小视差以避免近距离错误匹配
        stereo.setNumDisparities(448)
        stereo.setTextureThreshold(15)    # 适当提高纹理阈值以减少低纹理区域的错误匹配
        stereo.setUniquenessRatio(15)     # 增加唯一性比例以提高匹配质量
        stereo.setSpeckleWindowSize(25)   # 减小噪声窗口大小以更精确地过滤噪声
        stereo.setSpeckleRange(8)         # 减小噪声范围以更严格地过滤噪声
        stereo.setDisp12MaxDiff(1)        # 减小左右视差图最大差异容忍度以提高精度
        
        return stereo
    
    def rectify_images(self, imgL, imgR):
        """校正输入图像"""
        # 确保输入是灰度图
        if len(imgL.shape) > 2:
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        else:
            imgL_gray = imgL
        
        if len(imgR.shape) > 2:
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        else:
            imgR_gray = imgR
        
        # 应用重映射进行校正
        img1_rectified = cv2.remap(imgL_gray, self.camera_configs.left_map1, 
                                self.camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR_gray, self.camera_configs.right_map1, 
                                self.camera_configs.right_map2, cv2.INTER_LINEAR)
        
        return img1_rectified, img2_rectified
    
    def compute_disparity(self, rectified_left, rectified_right):
        """计算视差图"""
        disparity = self.stereo.compute(rectified_left, rectified_right)
        return disparity
    
    def normalize_disparity(self, disparity):
        """归一化视差图以便显示"""
        return cv2.normalize(disparity, disparity, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def compute_3d_points(self, disparity):
        """从视差图计算三维坐标"""
        # 创建一个副本以避免修改原始视差图
        disparity_copy = disparity.copy().astype(np.float32)
        
        # 3. 计算三维坐标
        threeD = cv2.reprojectImageTo3D(disparity_copy, self.camera_configs.Q, handleMissingValues=True)
        
        # 乘以16以得到正确的度量单位(mm)
        threeD = threeD *16.0
        
        return threeD
    
    def calculate_distance(self, threeD, x, y):
        """计算指定像素点的距离"""
        # 检查坐标是否有效
        h, w = threeD.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            # 获取三维坐标 (单位: mm)
            x_coord, y_coord, z_coord = threeD[y][x]
            
            # 计算距离 (单位: mm)
            distance = math.sqrt(x_coord**2 + y_coord**2 + z_coord**2)
            #distance = 252.22
            
            return {
                'x': x_coord,  # 单位: mm
                'y': y_coord,  # 单位: mm
                'z': z_coord,  # 单位: mm
                'distance': distance,   # 距离，单位: mm
                'valid': True
            }
        else:
            return None

# 主应用类
class BMRangingApp:
    def __init__(self):
        # 初始化相机参数
        self.camera_configs = CameraConfigs()
        # 初始化BM匹配器
        self.matcher = BlockMatcher(self.camera_configs)
        # 窗口名称
        self.WIN_NAME = 'BM Disparity Map'
        # 计数器用于保存图片
        self.counter = 0
        # 当前三维数据
        self.current_threeD = None
    
    def onmouse_pick_points(self, event, x, y, flags, param):
        """鼠标回调函数，用于点击获取三维坐标"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_threeD is not None:
            result = self.matcher.calculate_distance(self.current_threeD, x, y)
            if result:
                print(f'\n像素坐标: x = {x}, y = {y}')
                print(f'世界坐标 (mm): x = {result["x"]:.2f}, y = {result["y"]:.2f}, z = {result["z"]:.2f}')
                
                # 检查结果是否有效
                if 'valid' in result and not result['valid']:
                    print("警告: 距离计算无效，可能是因为视差计算不准确或匹配失败")
                else:
                    print(f'距离: {result["distance"]:.2f} mm')
    
    def process_frame(self, frame1, frame2):
        """处理一对双目图像"""
        import time
        start_time = time.time()

        # 校正图像
        rectified_left, rectified_right = self.matcher.rectify_images(frame1, frame2)

        # 计算视差
        disparity = self.matcher.compute_disparity(rectified_left, rectified_right)

        # 归一化视差图
        normalized_disp = self.matcher.normalize_disparity(disparity)

        # 转换为彩色以便显示
        disp_color = cv2.applyColorMap(normalized_disp, cv2.COLORMAP_JET)

        # 计算三维坐标
        threeD = self.matcher.compute_3d_points(disparity)
        self.current_threeD = threeD

        elapsed_time = time.time() - start_time
        print(f"\nBM处理时间: {elapsed_time:.4f} 秒")

        return rectified_left, rectified_right, disp_color
    
    def run_webcam(self):
        """从摄像头运行BM测距"""
        # 设置为可以支持4096x2160分辨率的摄像头，分割为两个1920x1080的图像
        cap = cv2.VideoCapture(0)
        cap.set(3, 3840)  # 设置宽度为3840，用于左右目各1920
        cap.set(4, 1080)  # 设置高度为1080
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)  # 使用WINDOW_NORMAL以便可以调整窗口大小
        cv2.resizeWindow(self.WIN_NAME, 1280, 720)  # 设置一个合理的窗口大小
        cv2.setMouseCallback(self.WIN_NAME, self.onmouse_pick_points)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取图像")
                    break
                
                # 分割左右目图像 - 针对1920x1080分辨率优化
                frame1 = frame[:, 0:1920]  # 左目
                frame2 = frame[:, 1920:3840]  # 右目
                
                # 处理图像
                rectified_left, rectified_right, disp_color = self.process_frame(frame1, frame2)
                
                # 调整显示尺寸以便更好地查看高分辨率图像
                display_size = (1280, 720)
                frame1_small = cv2.resize(frame1, display_size)
                rectified_left_small = cv2.resize(cv2.cvtColor(rectified_left, cv2.COLOR_GRAY2BGR), display_size)
                disp_color_small = cv2.resize(disp_color, display_size)
                
                # 显示结果
                cv2.imshow("Left Image", frame1_small)
                cv2.imshow("Rectified Left", rectified_left_small)
                cv2.imshow(self.WIN_NAME, disp_color_small)
                
                # 键盘交互
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前图像
                    now_time = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                    folder = os.getcwd()
                    path = os.path.join(folder, f"bm_result_{now_time}.jpg")
                    cv2.imwrite(path, disp_color)
                    print(f"结果已保存: {path}")
                    self.counter += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_from_images(self, left_path, right_path):
        """从图像文件运行BM测距"""
        print(f"加载图像: {left_path} 和 {right_path}")
        # 读取图像
        frame1 = cv2.imread(left_path)
        frame2 = cv2.imread(right_path)
        
        if frame1 is None or frame2 is None:
            print("无法读取图像文件")
            return
        
        print(f"原始图像尺寸: 左图 {frame1.shape}, 右图 {frame2.shape}")
        
        # 确保图像尺寸正确
        frame1 = cv2.resize(frame1, (1920, 1080))
        frame2 = cv2.resize(frame2, (1920, 1080))
        print(f"调整后图像尺寸: 左图 {frame1.shape}, 右图 {frame2.shape}")
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_NAME, 1280, 720)
        cv2.setMouseCallback(self.WIN_NAME, self.onmouse_pick_points)
        
        # 处理图像
        rectified_left, rectified_right, disp_color = self.process_frame(frame1, frame2)
        
        # 调整显示尺寸
        display_size = (1280, 720)
        frame1_small = cv2.resize(frame1, display_size)
        frame2_small = cv2.resize(frame2, display_size)
        rectified_left_small = cv2.resize(cv2.cvtColor(rectified_left, cv2.COLOR_GRAY2BGR), display_size)
        rectified_right_small = cv2.resize(cv2.cvtColor(rectified_right, cv2.COLOR_GRAY2BGR), display_size)
        disp_color_small = cv2.resize(disp_color, display_size)
        
        # 显示结果
        cv2.imshow("Left Image", frame1_small)
        cv2.imshow("Right Image", frame2_small)
        cv2.imshow("Rectified Left", rectified_left_small)
        cv2.imshow("Rectified Right", rectified_right_small)
        cv2.imshow(self.WIN_NAME, disp_color_small)
        
        print("按任意键退出，点击视差图查看对应点的三维坐标")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 运行演示
if __name__ == "__main__":
    app = BMRangingApp()
    
    # 选择运行模式
    # 1. 从摄像头运行
    #app.run_webcam()
    
    # 2. 从图像文件运行（取消下面三行的注释并提供正确的图像路径）
    # 先检查图像文件是否存在

    left_image_path = r"../0_image/100left/left_9.jpg"
    right_image_path = r"../0_image/100right/right_9.jpg"
    if os.path.exists(left_image_path) and os.path.exists(right_image_path):
        app.run_from_images(left_image_path, right_image_path)
    else:
        print(f"图像文件不存在，请检查路径: {left_image_path}, {right_image_path}")