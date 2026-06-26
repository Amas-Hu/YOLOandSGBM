import cv2
import numpy as np
import os
import json
import argparse
import glob

class CalibrationVerifier:
    """
    相机标定验证类
    用于验证相机标定结果的准确性
    """
    def __init__(self, calibration_file, square_size=0.03, pattern_size=(9, 6)):
        """
        初始化标定验证参数
        
        参数:
            calibration_file: 标定结果文件路径
            square_size: 棋盘格方块的实际尺寸（米）
            pattern_size: 棋盘格内角点数量，格式为 (宽度, 高度)
        """
        # 棋盘格参数
        self.square_size = square_size
        self.pattern_size = pattern_size
        
        # 相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        
        # 加载标定结果
        self.load_calibration_parameters(calibration_file)
    
    def load_calibration_parameters(self, calibration_file):
        """
        从文件加载标定参数
        
        参数:
            calibration_file: 标定结果文件路径
        """
        try:
            # 检查文件类型并加载
            file_ext = os.path.splitext(calibration_file)[1].lower()
            
            if file_ext == '.json':
                # 从JSON文件加载
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                
                self.image_size = tuple(data['image_size'])
                self.camera_matrix = np.array(data['camera_matrix'])
                self.dist_coeffs = np.array(data['dist_coeffs'])
                
                # 如果文件中包含棋盘格参数，则更新
                if 'square_size' in data:
                    self.square_size = data['square_size']
                if 'pattern_size' in data:
                    self.pattern_size = tuple(data['pattern_size'])
            
            elif file_ext == '.xml' or file_ext == '.yaml' or file_ext == '.yml':
                # 从OpenCV格式文件加载
                fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
                self.image_size = tuple(fs.getNode('image_size').mat())
                self.camera_matrix = fs.getNode('camera_matrix').mat()
                self.dist_coeffs = fs.getNode('dist_coeffs').mat()
                
                # 尝试加载棋盘格参数
                if not fs.getNode('square_size').empty():
                    self.square_size = fs.getNode('square_size').real()
                if not fs.getNode('pattern_size').empty():
                    self.pattern_size = tuple(fs.getNode('pattern_size').mat())
                fs.release()
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            print(f"成功加载标定参数")
            print(f"内参矩阵:\n{self.camera_matrix}")
            print(f"畸变系数:\n{self.dist_coeffs}")
            print(f"图像尺寸: {self.image_size}")
            
        except Exception as e:
            print(f"加载标定参数失败: {str(e)}")
            raise
    
    def _generate_object_points(self):
        """
        生成棋盘格的3D坐标点
        """
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # 乘以实际尺寸
        return objp
    
    def calculate_reprojection_error(self, image_file):
        """
        计算单张图像的重投影误差
        
        参数:
            image_file: 图像文件路径
        
        返回:
            重投影误差值
        """
        # 读取图像
        img = cv2.imread(image_file)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_file}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 生成3D坐标点
        objp = self._generate_object_points()
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_FAST_CHECK + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if not ret:
            print(f"在图像 {image_file} 中未找到棋盘格角点")
            return None
        
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 使用solvePnP获取旋转和平移向量
        ret, rvec, tvec = cv2.solvePnP(objp, corners_subpix, self.camera_matrix, self.dist_coeffs)
        
        if not ret:
            print(f"在图像 {image_file} 中求解PnP失败")
            return None
        
        # 重投影3D点到图像平面
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 计算重投影误差
        error = cv2.norm(corners_subpix, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        
        return error
    
    def verify_calibration(self, images_dir):
        """
        验证标定结果
        
        参数:
            images_dir: 包含验证图像的目录
        
        返回:
            平均重投影误差
        """
        # 检查标定参数是否已加载
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("未加载标定参数，请先调用load_calibration_parameters方法")
            return None
        
        # 获取目录中的所有图像文件
        image_files = []
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(images_dir, f'*.{ext.upper()}')))
        
        if not image_files:
            print(f"在目录 {images_dir} 中未找到图像文件")
            return None
        
        print(f"找到 {len(image_files)} 张验证图像")
        
        # 计算每张图像的重投影误差
        total_error = 0
        valid_images = 0
        
        for img_file in image_files:
            error = self.calculate_reprojection_error(img_file)
            if error is not None:
                total_error += error
                valid_images += 1
                print(f"图像 {os.path.basename(img_file)} 的重投影误差: {error:.6f}像素")
        
        if valid_images == 0:
            print("没有找到有效的验证图像")
            return None
        
        # 计算平均重投影误差
        avg_error = total_error / valid_images
        print(f"平均重投影误差: {avg_error:.6f}像素")
        
        return avg_error
    
    def visualize_calibration(self, image_file, output_file=None):
        """
        可视化标定结果
        
        参数:
            image_file: 图像文件路径
            output_file: 输出图像文件路径，若为None则显示图像
        
        返回:
            可视化后的图像
        """
        # 读取图像
        img = cv2.imread(image_file).copy()
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_file}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 生成3D坐标点
        objp = self._generate_object_points()
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_FAST_CHECK + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if not ret:
            print(f"在图像 {image_file} 中未找到棋盘格角点")
            return img
        
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 使用solvePnP获取旋转和平移向量
        ret, rvec, tvec = cv2.solvePnP(objp, corners_subpix, self.camera_matrix, self.dist_coeffs)
        
        if not ret:
            print(f"在图像 {image_file} 中求解PnP失败")
            return img
        
        # 绘制检测到的角点
        cv2.drawChessboardCorners(img, self.pattern_size, corners_subpix, ret)
        
        # 绘制3D坐标轴
        axis = np.float32([[3*self.square_size,0,0], [0,3*self.square_size,0], [0,0,-3*self.square_size]]).reshape(-1,3)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 获取棋盘格左上角点
        corner = tuple(corners_subpix[0].ravel())
        
        # 绘制坐标轴
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)  # X轴 (红色)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)  # Y轴 (绿色)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)  # Z轴 (蓝色)
        
        # 重投影3D点到图像平面
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 绘制重投影的点
        for i, pt in enumerate(imgpoints2):
            pt = tuple(pt.ravel())
            # 用不同颜色绘制重投影点和检测到的点
            cv2.circle(img, pt, 3, (0, 255, 255), -1)  # 黄色表示重投影点
        
        # 计算重投影误差
        error = cv2.norm(corners_subpix, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        
        # 在图像上显示重投影误差
        cv2.putText(img, f"重投影误差: {error:.6f}像素", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示或保存图像
        if output_file:
            cv2.imwrite(output_file, img)
            print(f"可视化结果已保存到 {output_file}")
        else:
            cv2.imshow('Calibration Verification', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img
    
    def undistort_and_compare(self, image_file, output_file=None):
        """
        去畸变并比较原始图像和去畸变后的图像
        
        参数:
            image_file: 图像文件路径
            output_file: 输出比较图像文件路径，若为None则显示图像
        
        返回:
            比较图像
        """
        # 读取图像
        img = cv2.imread(image_file)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_file}")
        
        # 获取图像尺寸
        h, w = img.shape[:2]
        
        # 计算优化后的相机矩阵
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 去畸变
        undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # 裁剪图像
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
        # 调整大小以匹配原始图像
        undistorted_resized = cv2.resize(undistorted, (w, h))
        
        # 创建比较图像
        comparison = np.hstack((img, undistorted_resized))
        
        # 在图像上添加标签
        cv2.putText(comparison, "原始图像", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, "去畸变后图像", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示或保存图像
        if output_file:
            cv2.imwrite(output_file, comparison)
            print(f"比较结果已保存到 {output_file}")
        else:
            # 创建一个可调整大小的窗口
            cv2.namedWindow('Original vs Undistorted', cv2.WINDOW_NORMAL)
            cv2.imshow('Original vs Undistorted', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return comparison

# 主程序
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='相机标定验证工具')
    parser.add_argument('--calibration_file', type=str, required=True, help='标定结果文件路径')
    parser.add_argument('--images_dir', type=str, help='验证图像目录')
    parser.add_argument('--image_file', type=str, help='单张验证图像文件路径')
    parser.add_argument('--output_dir', type=str, default='.', help='结果输出目录')
    parser.add_argument('--square_size', type=float, default=0.03, help='棋盘格方块尺寸（米）')
    parser.add_argument('--pattern_width', type=int, default=9, help='棋盘格宽度方向内角点数量')
    parser.add_argument('--pattern_height', type=int, default=6, help='棋盘格高度方向内角点数量')
    parser.add_argument('--visualize', action='store_true', help='可视化标定结果')
    parser.add_argument('--compare', action='store_true', help='比较原始图像和去畸变后的图像')
    
    args = parser.parse_args()
    
    # 创建标定验证实例
    verifier = CalibrationVerifier(
        args.calibration_file,
        square_size=args.square_size,
        pattern_size=(args.pattern_width, args.pattern_height)
    )
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 验证标定结果
    if args.images_dir:
        verifier.verify_calibration(args.images_dir)
    
    # 可视化标定结果
    if args.visualize and args.image_file:
        output_file = os.path.join(args.output_dir, 'calibration_visualization.png')
        verifier.visualize_calibration(args.image_file, output_file)
    
    # 比较原始图像和去畸变后的图像
    if args.compare and args.image_file:
        output_file = os.path.join(args.output_dir, 'undistortion_comparison.png')
        verifier.undistort_and_compare(args.image_file, output_file)