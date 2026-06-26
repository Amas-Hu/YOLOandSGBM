import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random

# ================== 参数配置 ==================
# IMG_SIZE 和 SUB_SIZE 将根据输入图像动态计算

# ================== 图像增强函数 ==================
def strong_augment(img):
    """
    对单张图像进行随机增强：尺度缩放、水平翻转、颜色扰动
    """
    # 随机缩放
    scale = np.random.uniform(0.4, 1.0)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 随机水平翻转
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # HSV 颜色扰动
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= np.random.uniform(0.5, 1.5)  # 饱和度
    hsv[..., 2] *= np.random.uniform(0.5, 1.5)  # 亮度
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

# ================== Mosaic 拼接函数 ==================
def mosaic_strong(images):
    """
    对 1 张或 4 张图像进行强增强 Mosaic 拼接
    """
    # 动态计算输出大小：取所有图像的最大边长，确保是偶数
    max_size = 0
    for img in images:
        h, w = img.shape[:2]
        max_size = max(max_size, h, w)
    
    # 确保输出大小是偶数，方便分割为4个象限
    IMG_SIZE = max_size + (max_size % 2)
    SUB_SIZE = IMG_SIZE // 2  # 每张图像缩放后的大小
    
    mosaic = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)

    # 处理单张图片情况：生成 4 张不同增强版本
    if len(images) == 1:
        base_img = images[0]
        # 对单张图片进行 4 次不同的增强
        resized_imgs = [cv2.resize(strong_augment(base_img), (SUB_SIZE, SUB_SIZE)) for _ in range(4)]
    else:  # 处理 4 张图片情况
        # 强增强 + 缩放
        resized_imgs = [cv2.resize(strong_augment(img), (SUB_SIZE, SUB_SIZE)) for img in images]

    # 四象限拼接
    mosaic[0:SUB_SIZE, 0:SUB_SIZE] = resized_imgs[0]          # 左上
    mosaic[0:SUB_SIZE, SUB_SIZE:IMG_SIZE] = resized_imgs[1]   # 右上
    mosaic[SUB_SIZE:IMG_SIZE, 0:SUB_SIZE] = resized_imgs[2]   # 左下
    mosaic[SUB_SIZE:IMG_SIZE, SUB_SIZE:IMG_SIZE] = resized_imgs[3]  # 右下

    # 绘制白色十字分割线，使 Mosaic 结构更明显
    cv2.line(mosaic, (SUB_SIZE, 0), (SUB_SIZE, IMG_SIZE), (255, 255, 255), 3)
    cv2.line(mosaic, (0, SUB_SIZE), (IMG_SIZE, SUB_SIZE), (255, 255, 255), 3)

    return mosaic

# ================== 前端 GUI ==================
class MosaicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("强增强 Mosaic 数据增强演示 - 对比显示")
        self.images = []
        self.original_img = None  # 保存原始图片数据，用于保存
        self.mosaic_img = None  # 保存增强后的图片数据，用于保存

        # 按钮：选择图片
        tk.Button(root, text="选择 1 张或 4 张图片", command=self.load_images,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)

        # 按钮：执行 Mosaic
        tk.Button(root, text="执行 强 Mosaic 增强", command=self.run_mosaic,
                  width=25, font=("微软雅黑", 11)).pack(pady=5)

        # 按钮：保存对比结果
        tk.Button(root, text="保存对比结果", command=self.save_comparison,
                  width=25, font=("微软雅黑", 11)).pack(pady=5)

        # 对比显示框架
        self.compare_frame = tk.Frame(root)
        self.compare_frame.pack(pady=10)

        # 原始图片显示区域
        self.original_panel = tk.Label(self.compare_frame, text="原始图片")
        self.original_panel.grid(row=0, column=0, padx=10)

        # 增强后图片显示区域
        self.augmented_panel = tk.Label(self.compare_frame, text="增强后 Mosaic")
        self.augmented_panel.grid(row=0, column=1, padx=10)

        # 实际图片显示
        self.original_img_label = tk.Label(self.compare_frame)
        self.original_img_label.grid(row=1, column=0, padx=10)

        self.augmented_img_label = tk.Label(self.compare_frame)
        self.augmented_img_label.grid(row=1, column=1, padx=10)

    def load_images(self):
        """
        加载用户选择的 1 张或 4 张图片
        """
        paths = filedialog.askopenfilenames(
            title="请选择 1 张或 4 张图片",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if len(paths) not in [1, 4]:
            messagebox.showerror("错误", "必须选择 1 张或 4 张图片")
            return
        self.images = [cv2.imread(p) for p in paths]
        messagebox.showinfo("成功", f"图片加载完成，共 {len(self.images)} 张")

    def run_mosaic(self):
        """
        执行 Mosaic 数据增强并显示对比
        """
        if len(self.images) not in [1, 4]:
            messagebox.showerror("错误", "请先选择 1 张或 4 张图片")
            return

        # 执行马赛克增强
        mosaic_img = mosaic_strong(self.images)
        mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)
        
        # 处理原始图片显示
        if len(self.images) == 1:
            # 单张图片情况：直接显示原始图片
            original_img = cv2.cvtColor(self.images[0], cv2.COLOR_BGR2RGB)
        else:
            # 4张图片情况：将4张原始图片拼接成与马赛克相同的布局
            # 动态计算大小
            max_size = 0
            for img in self.images:
                h, w = img.shape[:2]
                max_size = max(max_size, h, w)
            IMG_SIZE = max_size + (max_size % 2)  # 确保是偶数
            SUB_SIZE = IMG_SIZE // 2
            
            original_mosaic = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
            resized_originals = [cv2.resize(img, (SUB_SIZE, SUB_SIZE)) for img in self.images]
            original_mosaic[0:SUB_SIZE, 0:SUB_SIZE] = resized_originals[0]          # 左上
            original_mosaic[0:SUB_SIZE, SUB_SIZE:IMG_SIZE] = resized_originals[1]   # 右上
            original_mosaic[SUB_SIZE:IMG_SIZE, 0:SUB_SIZE] = resized_originals[2]   # 左下
            original_mosaic[SUB_SIZE:IMG_SIZE, SUB_SIZE:IMG_SIZE] = resized_originals[3]  # 右下
            # 绘制白色十字分割线
            cv2.line(original_mosaic, (SUB_SIZE, 0), (SUB_SIZE, IMG_SIZE), (255, 255, 255), 3)
            cv2.line(original_mosaic, (0, SUB_SIZE), (IMG_SIZE, SUB_SIZE), (255, 255, 255), 3)
            original_img = cv2.cvtColor(original_mosaic, cv2.COLOR_BGR2RGB)
        
        # 保存原始图片和增强后的图片数据，用于保存
        self.original_img = original_img
        self.mosaic_img = mosaic_img
        
        # 调整图片大小并转换为 PhotoImage，统一显示为 420x420
        original_img_pil = Image.fromarray(original_img).resize((420, 420))
        original_img_tk = ImageTk.PhotoImage(original_img_pil)
        
        mosaic_img_pil = Image.fromarray(mosaic_img).resize((420, 420))
        mosaic_img_tk = ImageTk.PhotoImage(mosaic_img_pil)
        
        # 更新显示
        self.original_img_label.config(image=original_img_tk)
        self.original_img_label.image = original_img_tk
        
        self.augmented_img_label.config(image=mosaic_img_tk)
        self.augmented_img_label.image = mosaic_img_tk

    def save_comparison(self):
        """
        保存增强前后的对比结果
        """
        if self.original_img is None or self.mosaic_img is None:
            messagebox.showerror("错误", "请先执行 Mosaic 增强")
            return
        
        # 弹出文件对话框选择保存路径
        save_path = filedialog.asksaveasfilename(
            title="保存对比结果",
            defaultextension=".jpg",
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png"), ("所有文件", "*.*")]
        )
        
        if not save_path:
            return  # 用户取消了保存
        
        try:
            # 确保两张图片大小相同，便于拼接
            h1, w1 = self.original_img.shape[:2]
            h2, w2 = self.mosaic_img.shape[:2]
            
            # 取最大高度，统一宽度
            max_h = max(h1, h2)
            
            # 调整原始图片大小
            original_resized = cv2.resize(self.original_img, (w1, max_h))
            # 调整增强后图片大小
            mosaic_resized = cv2.resize(self.mosaic_img, (w2, max_h))
            
            # 拼接两张图片，中间留10像素空白
            blank = np.ones((max_h, 10, 3), dtype=np.uint8) * 255
            comparison = np.hstack((original_resized, blank, mosaic_resized))
            
            # 转换为BGR格式保存
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, comparison_bgr)
            
            messagebox.showinfo("成功", f"对比结果已保存到 {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{str(e)}")

# ================== 主程序 ==================
if __name__ == "__main__":
    root = tk.Tk()
    MosaicGUI(root)
    root.mainloop()
