import cv2
import numpy as np
import os
import json
import argparse
import glob
import time

class CameraCalibrator:
    """
    相机标定类
    用于计算相机的内参矩阵、畸变系数以及立体相机的外参
    """
    def __init__(self, square_size=0.03, pattern_size=(9, 6), 
                 calibration_images_dir=None, output_dir=".", debug=False):
        """
        初始化相机标定参数
        
        参数:
            square_size: 棋盘格方块的实际尺寸（米）
            pattern_size: 棋盘格内角点数量，格式为 (宽度, 高度)
            calibration_images_dir: 标定图像目录路径
            output_dir: 标定结果输出目录
            debug: 是否启用调试模式，显示角点检测结果
        """
        # 棋盘格参数
        self.square_size = square_size  # 米
        self.pattern_size = pattern_size  # 内角点数量
        
        # 图像和输出目录
        self.calibration_images_dir = calibration_images_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 调试模式
        self.debug = debug
        
        # 相机参数
        self.camera_matrix = None  # 内参矩阵
        self.dist_coeffs = None  # 畸变系数
        self.rvecs = None  # 旋转向量
        self.tvecs = None  # 平移向量
        
        # 重投影误差
        self.reprojection_error = None
        
        # 3D点和2D点
        self.object_points = []  # 3D世界坐标点
        self.image_points = []  # 2D图像坐标点
        
        # 图像尺寸
        self.image_size = None
    
    def _generate_object_points(self):
        """
        生成棋盘格的3D坐标点
        """
        # 创建棋盘格的3D坐标点
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # 乘以实际尺寸
        return objp
    
    def find_chessboard_corners(self, images=None):
        """
        寻找标定图像中的棋盘格角点
        
        参数:
            images: 可选，手动提供图像列表，若不提供则从calibration_images_dir读取
        
        返回:
            成功检测到角点的图像数量
        """
        # 初始化计数器
        success_count = 0
        
        # 生成3D坐标点
        objp = self._generate_object_points()
        
        # 读取图像
        image_files = []
        if images is not None:
            # 使用提供的图像列表
            image_files = images
        elif self.calibration_images_dir is not None:
            # 从目录读取图像
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(self.calibration_images_dir, f'*.{ext}')))
                image_files.extend(glob.glob(os.path.join(self.calibration_images_dir, f'*.{ext.upper()}')))
        
        if not image_files:
            print("未找到标定图像")
            return 0
        
        print(f"找到 {len(image_files)} 张标定图像")
        
        # 遍历所有图像
        for i, img_file in enumerate(image_files):
            # 读取图像
            img = cv2.imread(img_file)
            if img is None:
                print(f"无法读取图像: {img_file}")
                continue
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 存储图像尺寸
            if self.image_size is None:
                self.image_size = gray.shape[::-1]  # (width, height)
            
            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_FAST_CHECK + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # 如果找到角点，进行亚像素精确化
            if ret:
                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 保存点
                self.object_points.append(objp)
                self.image_points.append(corners_subpix)
                
                success_count += 1
                print(f"成功检测到图像 {i+1}/{len(image_files)} 的角点")
                
                # 调试模式下显示角点
                if self.debug:
                    # 绘制角点
                    img_corners = cv2.drawChessboardCorners(img, self.pattern_size, corners_subpix, ret)
                    
                    # 显示图像
                    cv2.imshow('Chessboard Corners', img_corners)
                    cv2.waitKey(500)  # 显示0.5秒
        
        # 关闭所有窗口
        if self.debug:
            cv2.destroyAllWindows()
        
        print(f"共成功检测到 {success_count} 张图像的角点")
        return success_count
    
    def calibrate(self):
        """
        执行相机标定
        
        返回:
            标定是否成功
        """
        if not self.object_points or not self.image_points:
            print("没有足够的角点数据进行标定，请先调用find_chessboard_corners方法")
            return False
        
        print("开始相机标定...")
        start_time = time.time()
        
        # 执行相机标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, None, None
        )
        
        if not ret:
            print("相机标定失败")
            return False
        
        # 计算重投影误差
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(self.object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        reprojection_error = total_error / len(self.object_points)
        
        # 保存标定结果
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = reprojection_error
        
        # 打印标定结果
        print(f"相机标定完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"重投影误差: {reprojection_error:.6f}像素")
        print(f"内参矩阵:\n{camera_matrix}")
        print(f"畸变系数:\n{dist_coeffs}")
        
        return True
    
    def save_calibration_results(self, filename='calibration.json'):
        """
        保存标定结果到文件
        
        参数:
            filename: 保存的文件名
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("没有可用的标定结果，请先执行标定")
            return False
        
        try:
            # 构建标定结果字典
            calibration_data = {
                'image_size': self.image_size,
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'reprojection_error': float(self.reprojection_error),
                'square_size': self.square_size,
                'pattern_size': self.pattern_size,
                'calibration_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存为JSON文件
            file_path = os.path.join(self.output_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
            print(f"标定结果已保存到 {file_path}")
            
            # 同时保存为OpenCV的XML格式
            xml_file = os.path.splitext(file_path)[0] + '.xml'
            fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_WRITE)
            fs.write('image_size', self.image_size)
            fs.write('camera_matrix', self.camera_matrix)
            fs.write('dist_coeffs', self.dist_coeffs)
            fs.write('reprojection_error', self.reprojection_error)
            fs.write('square_size', self.square_size)
            fs.release()
            
            print(f"标定结果已保存到 {xml_file}")
            
            # 保存标定结果摘要文本文件
            txt_file = os.path.splitext(file_path)[0] + '.txt'
            with open(txt_file, 'w') as f:
                f.write(f"相机标定结果\n")
                f.write(f"标定时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"图像尺寸: {self.image_size}\n")
                f.write(f"重投影误差: {self.reprojection_error:.6f}像素\n")
                f.write(f"内参矩阵:\n{self.camera_matrix}\n")
                f.write(f"畸变系数:\n{self.dist_coeffs}\n")
                f.write(f"棋盘格尺寸: {self.square_size}米\n")
                f.write(f"棋盘格内角点数量: {self.pattern_size}\n")
            
            print(f"标定结果摘要已保存到 {txt_file}")
            
            return True
        except Exception as e:
            print(f"保存标定结果失败: {str(e)}")
            return False
    
    def load_calibration_results(self, filename='calibration.json'):
        """
        从文件加载标定结果
        
        参数:
            filename: 标定结果文件名
        """
        try:
            file_path = os.path.join(self.output_dir, filename)
            
            # 尝试从JSON文件加载
            if os.path.exists(file_path) and file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    calibration_data = json.load(f)
                
                self.image_size = tuple(calibration_data['image_size'])
                self.camera_matrix = np.array(calibration_data['camera_matrix'])
                self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
                self.reprojection_error = calibration_data.get('reprojection_error', None)
                self.square_size = calibration_data.get('square_size', self.square_size)
                self.pattern_size = tuple(calibration_data.get('pattern_size', self.pattern_size))
            
            # 尝试从XML文件加载
            elif os.path.exists(os.path.splitext(file_path)[0] + '.xml'):
                xml_file = os.path.splitext(file_path)[0] + '.xml'
                fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
                self.image_size = tuple(fs.getNode('image_size').mat())
                self.camera_matrix = fs.getNode('camera_matrix').mat()
                self.dist_coeffs = fs.getNode('dist_coeffs').mat()
                self.reprojection_error = fs.getNode('reprojection_error').real()
                if fs.getNode('square_size').empty() is False:
                    self.square_size = fs.getNode('square_size').real()
                if fs.getNode('pattern_size').empty() is False:
                    self.pattern_size = tuple(fs.getNode('pattern_size').mat())
                fs.release()
            else:
                print(f"未找到标定结果文件: {file_path}")
                return False
            
            print(f"成功加载标定结果")
            print(f"图像尺寸: {self.image_size}")
            print(f"内参矩阵:\n{self.camera_matrix}")
            print(f"畸变系数:\n{self.dist_coeffs}")
            if self.reprojection_error is not None:
                print(f"重投影误差: {self.reprojection_error:.6f}像素")
            
            return True
        except Exception as e:
            print(f"加载标定结果失败: {str(e)}")
            return False
    
    def undistort_image(self, image):
        """
        使用标定结果对图像进行去畸变
        
        参数:
            image: 输入图像
        
        返回:
            去畸变后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("没有可用的标定结果，无法进行图像去畸变")
            return image
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 计算优化后的相机矩阵
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 去畸变
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # 裁剪图像（如果需要）
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def run_full_calibration(self):
        """
        运行完整的标定流程
        
        返回:
            标定是否成功
        """
        # 1. 寻找棋盘格角点
        success_count = self.find_chessboard_corners()
        if success_count < 5:  # 至少需要5张成功的标定图像
            print("成功检测到的角点图像数量不足，无法完成标定")
            return False
        
        # 2. 执行标定
        if not self.calibrate():
            return False
        
        # 3. 保存标定结果
        return self.save_calibration_results()

# 主程序
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='相机标定工具')
    parser.add_argument('--images_dir', type=str, required=True, help='标定图像目录')
    parser.add_argument('--output_dir', type=str, default='.', help='标定结果输出目录')
    parser.add_argument('--square_size', type=float, default=0.03, help='棋盘格方块尺寸（米）')
    parser.add_argument('--pattern_width', type=int, default=9, help='棋盘格宽度方向内角点数量')
    parser.add_argument('--pattern_height', type=int, default=6, help='棋盘格高度方向内角点数量')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--load', type=str, help='加载已有的标定结果文件')
    
    args = parser.parse_args()
    
    # 创建相机标定实例
    calibrator = CameraCalibrator(
        square_size=args.square_size,
        pattern_size=(args.pattern_width, args.pattern_height),
        calibration_images_dir=args.images_dir,
        output_dir=args.output_dir,
        debug=args.debug
    )
    
    if args.load:
        # 加载已有的标定结果
        calibrator.load_calibration_results(args.load)
    else:
        # 执行完整的标定流程
        success = calibrator.run_full_calibration()
        if success:
            print("相机标定完成！")
        else:
            print("相机标定失败！")