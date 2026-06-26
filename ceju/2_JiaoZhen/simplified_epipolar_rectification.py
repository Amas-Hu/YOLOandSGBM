import numpy as np
import cv2
import numpy as np
import os

class SimplifiedEpipolarRectifier:
    """
    简化版极线矫正工具类
    """
    def __init__(self):
        """
        初始化极线矫正器，设置相机参数
        """
        # 相机参数（使用实际的标定参数）
        self.left_camera_matrix = np.array([[1.057389594483230e+03, 1.698879225510185, 5.800114785214288e+02],
                                        [0, 1.058384060607384e+03, 5.084522757265003e+02],
                                        [0, 0, 1]])

        self.distortion0 = np.array([[0.048033739382123, 0.311433858924166, 0.001205307356943, 0.001270572815158, -0.385926882800597]])

        self.right_camera_matrix = np.array([[1.065085617396094e+03, 1.384118782509674, 5.809912806789391e+02],
                                            [0, 1.065864639143073e+03, 4.958062841067978e+02],
                                            [0, 0, 1]])

        self.distortion1 = np.array([[-0.049056548243087, 0.345747823496975, 0.001058038492965, 0.001167808554822, -0.461392018886485]])

        self.R = np.array([[1, 0.001101971133142, 8.064727977011493e-04],
                        [-0.001098406406917, 1, -0.004407319052475],
                        [-8.113212168754618e-04, 0.004406429108336, 1]])

        self.T = np.array([[-61.340492524588896], [-0.114434490871765], [1.758889832299615]])

        # 初始化映射表
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None
        self.image_size = None

    def compute_rectification_maps(self, width, height):
        """
        计算极线矫正映射表
        """
        self.image_size = (width, height)
        
        # 计算旋转矩阵和投影矩阵
        R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = \
            cv2.stereoRectify(self.left_camera_matrix, self.distortion0,\
                            self.right_camera_matrix, self.distortion1,\
                            np.array([width, height]), self.R, self.T)
        
        # 计算映射表
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.distortion0, R_l, P_l, 
            (width, height), cv2.CV_32FC1)
            
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.distortion1, R_r, P_r, 
            (width, height), cv2.CV_32FC1)

    def split_image(self, stereo_image):
        """
        分割图像为左右两部分
        """
        h, w = stereo_image.shape[:2]
        # 从中间分割图像
        mid_point = w // 2
        left_image = stereo_image[:, :1280]  # 使用固定宽度分割
        right_image = stereo_image[:, 1280:2560]
        
        return left_image, right_image

    def undistort_and_remap(self, image, map1, map2):
        """
        应用映射表进行图像矫正
        """
        return cv2.remap(image, map1, map2, cv2.INTER_CUBIC)

    def generate_comparison_image(self, left_image, right_image, rectified_left, rectified_right):
        """
        生成原始与矫正图像对比图
        """
        # 确保所有图像都是彩色的
        def ensure_color(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        
        left_image = ensure_color(left_image)
        right_image = ensure_color(right_image)
        rectified_left = ensure_color(rectified_left)
        rectified_right = ensure_color(rectified_right)
        
        # 获取图像尺寸
        h, w = left_image.shape[:2]
        
        # 创建对比图像
        top_row = np.hstack((left_image, right_image))
        bottom_row = np.hstack((rectified_left, rectified_right))
        comparison_image = np.vstack((top_row, bottom_row))
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_image, "Original Left", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison_image, "Original Right", (w + 10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison_image, "Rectified Left", (10, h + 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison_image, "Rectified Right", (w + 10, h + 30), font, 0.7, (0, 255, 0), 2)
        
        return comparison_image

    def draw_epipolar_lines(self, left_image, right_image, num_lines=10):
        """
        在左右图像上绘制极线
        """
        # 复制图像
        left_with_lines = left_image.copy()
        right_with_lines = right_image.copy()
        
        # 获取图像尺寸
        h, w = left_image.shape[:2]
        
        # 计算极线间隔
        line_interval = h // (num_lines + 1)
        
        # 绘制极线
        for i in range(1, num_lines + 1):
            y = i * line_interval
            cv2.line(left_with_lines, (0, y), (w, y), (0, 255, 0), 1)
            cv2.line(right_with_lines, (0, y), (w, y), (0, 255, 0), 1)
        
        return left_with_lines, right_with_lines

    def process_image(self):
        """
        处理图像并保存结果
        """
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建图像路径
        image_path = os.path.join(current_dir, "251.jpg")
        
        # 读取图像
        img = None
        
        # 尝试读取251.jpg
        if os.path.exists(image_path):
            print(f"正在读取图像: {image_path}")
            img = cv2.imread(image_path)
        
        # 如果251.jpg无法读取，尝试使用测试图片
        if img is None:
            print(f"警告: 无法读取或找不到251.jpg，尝试使用测试图片...")
            
            # 尝试使用ceju/images目录下的测试图片
            test_left_path = os.path.join(current_dir, "..", "images", "left", "left_0.jpg")
            test_right_path = os.path.join(current_dir, "..", "images", "right", "right_0.jpg")
            
            if os.path.exists(test_left_path) and os.path.exists(test_right_path):
                print(f"尝试使用测试图片: {test_left_path} 和 {test_right_path}")
                
                # 读取左右测试图像
                left_image = cv2.imread(test_left_path)
                right_image = cv2.imread(test_right_path)
                
                if left_image is not None and right_image is not None:
                    print("Successfully loaded test images")
                    # Directly process test images without splitting
                    return self.process_test_images(left_image, right_image)
                else:
                    print("Error: Failed to read test images")
                    return False
            else:
                print(f"Error: Cannot find test images at paths: {test_left_path} and {test_right_path}")
                print("Please ensure that 251.jpg exists and is readable, or that there are test images in the ceju/images directory")
                return False
        
        # Split image
        print("Splitting image...")
        left_image, right_image = self.split_image(img)
        
        # Get image dimensions
        h, w = left_image.shape[:2]
        
        # Compute rectification maps
        print("Computing epipolar rectification maps...")
        self.compute_rectification_maps(w, h)
        
        # Rectify images
        print("Performing epipolar rectification...")
        rectified_left = self.undistort_and_remap(left_image, self.left_map1, self.left_map2)
        rectified_right = self.undistort_and_remap(right_image, self.right_map1, self.right_map2)
        
        # Generate comparison image
        print("Generating comparison image...")
        comparison_image = self.generate_comparison_image(left_image, right_image, rectified_left, rectified_right)
        
        # Draw epipolar lines
        left_with_lines, right_with_lines = self.draw_epipolar_lines(rectified_left, rectified_right)
        
        # Save results
        cv2.imwrite(os.path.join(current_dir, "rectification_comparison.png"), comparison_image)
        cv2.imwrite(os.path.join(current_dir, "rectified_left.png"), rectified_left)
        cv2.imwrite(os.path.join(current_dir, "rectified_right.png"), rectified_right)
        cv2.imwrite(os.path.join(current_dir, "rectified_left_with_lines.png"), left_with_lines)
        cv2.imwrite(os.path.join(current_dir, "rectified_right_with_lines.png"), right_with_lines)
        
        print("Epipolar rectification completed, results saved!")
        print("Saved files:")
        print("- rectification_comparison.png: Comparison of original and rectified images")
        print("- rectified_left.png: Rectified left image")
        print("- rectified_right.png: Rectified right image")
        print("- rectified_left_with_lines.png: Left image with epipolar lines")
        print("- rectified_right_with_lines.png: Right image with epipolar lines")
        
        # Display results
        cv2.namedWindow("Comparison of epipolar correction results", cv2.WINDOW_NORMAL)
        cv2.imshow("Comparison of epipolar correction results", comparison_image)
        
        cv2.namedWindow("Left image (with epipolar lines)", cv2.WINDOW_NORMAL)
        cv2.imshow("Left image (with epipolar lines)", left_with_lines)
        
        cv2.namedWindow("Right image (with epipolar lines)", cv2.WINDOW_NORMAL)
        cv2.imshow("Right image (with epipolar lines)", right_with_lines)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    def process_test_images(self, left_image, right_image):
        """
        处理测试图像
        """
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取图像尺寸
        h, w = left_image.shape[:2]
        
        # 计算矫正映射表
        print("正在计算极线矫正映射表...")
        self.compute_rectification_maps(w, h)
        
        # 矫正图像
        print("正在进行极线矫正...")
        rectified_left = self.undistort_and_remap(left_image, self.left_map1, self.left_map2)
        rectified_right = self.undistort_and_remap(right_image, self.right_map1, self.right_map2)
        
        # 生成对比图像
        print("正在生成对比图像...")
        comparison_image = self.generate_comparison_image(left_image, right_image, rectified_left, rectified_right)
        
        # 绘制极线
        left_with_lines, right_with_lines = self.draw_epipolar_lines(rectified_left, rectified_right)
        
        # 保存结果
        cv2.imwrite(os.path.join(current_dir, "rectification_comparison.png"), comparison_image)
        cv2.imwrite(os.path.join(current_dir, "rectified_left.png"), rectified_left)
        cv2.imwrite(os.path.join(current_dir, "rectified_right.png"), rectified_right)
        cv2.imwrite(os.path.join(current_dir, "rectified_left_with_lines.png"), left_with_lines)
        cv2.imwrite(os.path.join(current_dir, "rectified_right_with_lines.png"), right_with_lines)
        
        print("Epipolar rectification completed, results saved!")
        print("Note: Using test images instead of 251.jpg")
        print("Saved files:")
        print("- rectification_comparison.png: Comparison of original and rectified images")
        print("- rectified_left.png: Rectified left image")
        print("- rectified_right.png: Rectified right image")
        print("- rectified_left_with_lines.png: Left image with epipolar lines")
        print("- rectified_right_with_lines.png: Right image with epipolar lines")
        
        # 显示结果
        cv2.namedWindow("Comparison of epipolar correction results", cv2.WINDOW_NORMAL)
        cv2.imshow("Comparison of epipolar correction results", comparison_image)
        
        cv2.namedWindow("Left image (with epipolar lines)", cv2.WINDOW_NORMAL)
        cv2.imshow("Left image (with epipolar lines)", left_with_lines)
        
        cv2.namedWindow("Right image (with epipolar lines)", cv2.WINDOW_NORMAL)
        cv2.imshow("Right image (with epipolar lines)", right_with_lines)
        
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True

# 主程序
if __name__ == "__main__":
    print("======= Epipolar Rectification Tool =======")
    print("Processing 251.jpg directly, no command-line arguments needed")
    
    # Create epipolar rectifier instance
    rectifier = SimplifiedEpipolarRectifier()
    
    # Process image
    success = rectifier.process_image()
    
    if not success:
        print("Processing failed!")
    else:
        print("Processing successful!")