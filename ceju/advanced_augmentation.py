import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import os

# ================== 数据增强函数 ==================

# 1. 色彩空间增强
def color_space_augment(img):
    """
    色彩空间增强：随机调整亮度、对比度、饱和度和色调
    """
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 随机调整
    hsv[..., 0] *= random.uniform(0.8, 1.2)  # 色调
    hsv[..., 1] *= random.uniform(0.5, 1.5)  # 饱和度
    hsv[..., 2] *= random.uniform(0.5, 1.5)  # 亮度
    
    # 确保值在有效范围内
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 2. 旋转增强
def rotate_augment(img):
    """
    旋转增强：随机旋转图片
    """
    h, w = img.shape[:2]
    # 随机旋转角度（-45到45度）
    angle = random.uniform(-45, 45)
    
    # 计算旋转矩阵
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 执行旋转
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 3. 缩放增强
def scale_augment(img):
    """
    缩放增强：随机缩放图片
    """
    h, w = img.shape[:2]
    # 随机缩放因子（0.5到1.5）
    scale = random.uniform(0.5, 1.5)
    
    # 计算新的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 执行缩放
    resized = cv2.resize(img, (new_w, new_h))
    
    # 如果缩放后的图片比原图片大，随机裁剪回原尺寸
    if scale > 1.0:
        x = random.randint(0, new_w - w)
        y = random.randint(0, new_h - h)
        resized = resized[y:y+h, x:x+w]
    # 如果缩放后的图片比原图片小，居中填充回原尺寸
    else:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        resized = cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, 
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return resized

# 4. 透视增强
def perspective_augment(img):
    """
    透视增强：随机透视变换
    """
    h, w = img.shape[:2]
    
    # 原始四个角点
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 随机扰动四个角点
    offset = random.randint(0, int(min(w, h) * 0.1))
    dst_points = np.float32([
        [random.randint(0, offset), random.randint(0, offset)],
        [w - random.randint(0, offset), random.randint(0, offset)],
        [w - random.randint(0, offset), h - random.randint(0, offset)],
        [random.randint(0, offset), h - random.randint(0, offset)]
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 执行透视变换
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 5. 上下翻转增强
def flip_vertical_augment(img):
    """
    上下翻转增强
    """
    return cv2.flip(img, 0)

# 6. 左右翻转增强
def flip_horizontal_augment(img):
    """
    左右翻转增强
    """
    return cv2.flip(img, 1)

# 7. BGR通道交换增强
def channel_swap_augment(img):
    """
    BGR通道交换增强：随机交换BGR通道
    """
    # 随机生成通道顺序
    channels = [0, 1, 2]
    random.shuffle(channels)
    
    # 交换通道
    return img[..., channels]

# 8. Mosaic增强
def mosaic_augment(img, other_imgs=None):
    """
    Mosaic增强：将4张图片拼接成马赛克
    """
    h, w = img.shape[:2]
    mosaic_size = max(h, w)
    
    # 创建马赛克画布
    mosaic = np.full((mosaic_size, mosaic_size, 3), 114, dtype=np.uint8)
    
    # 如果没有提供其他图片，使用原图生成3张增强后的图片
    if other_imgs is None:
        other_imgs = [color_space_augment(img), rotate_augment(img), scale_augment(img)]
    
    # 确保有3张其他图片
    while len(other_imgs) < 3:
        other_imgs.append(color_space_augment(img))
    
    # 选择4张图片（包括原图）
    imgs = [img] + other_imgs[:3]
    random.shuffle(imgs)
    
    # 计算每张图片的大小
    sub_size = mosaic_size // 2
    
    # 缩放并粘贴图片到马赛克画布
    for i in range(4):
        # 缩放图片
        resized = cv2.resize(imgs[i], (sub_size, sub_size))
        
        # 计算位置
        x = (i % 2) * sub_size
        y = (i // 2) * sub_size
        
        # 粘贴图片
        mosaic[y:y+sub_size, x:x+sub_size] = resized
    
    # 调整回原尺寸
    return cv2.resize(mosaic, (w, h))

# 9. Mixup增强
def mixup_augment(img, other_img=None):
    """
    Mixup增强：将两张图片按比例混合
    """
    h, w = img.shape[:2]
    
    # 如果没有提供其他图片，使用原图生成一张增强后的图片
    if other_img is None:
        other_img = color_space_augment(img)
    
    # 调整其他图片的尺寸
    other_img = cv2.resize(other_img, (w, h))
    
    # 随机混合比例
    alpha = random.uniform(0.2, 0.8)
    
    # 混合图片
    mixed = cv2.addWeighted(img, alpha, other_img, 1 - alpha, 0)
    
    return mixed.astype(np.uint8)

# 10. Cutmix增强
def cutmix_augment(img, other_img=None):
    """
    Cutmix增强：将另一张图片的一部分裁剪并粘贴到原图
    """
    h, w = img.shape[:2]
    
    # 如果没有提供其他图片，使用原图生成一张增强后的图片
    if other_img is None:
        other_img = color_space_augment(img)
    
    # 调整其他图片的尺寸
    other_img = cv2.resize(other_img, (w, h))
    
    # 随机生成裁剪区域
    cut_size = int(min(w, h) * random.uniform(0.3, 0.7))
    x1 = random.randint(0, w - cut_size)
    y1 = random.randint(0, h - cut_size)
    x2 = x1 + cut_size
    y2 = y1 + cut_size
    
    # 裁剪并粘贴
    result = img.copy()
    result[y1:y2, x1:x2] = other_img[y1:y2, x1:x2]
    
    return result

# 11. 平移增强
def translate_augment(img):
    """
    平移增强：随机平移图片
    """
    h, w = img.shape[:2]
    
    # 随机平移距离（-10%到10%）
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    
    # 计算平移矩阵
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # 执行平移
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 12. 剪切增强
def shear_augment(img):
    """
    剪切增强：随机剪切图片
    """
    h, w = img.shape[:2]
    
    # 随机剪切因子（-0.2到0.2）
    shear_x = random.uniform(-0.2, 0.2)
    shear_y = random.uniform(-0.2, 0.2)
    
    # 计算剪切矩阵
    matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    
    # 执行剪切
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

# 所有增强方法的列表和名称
AUGMENTATIONS = {
    "色彩空间增强": color_space_augment,
    "旋转增强": rotate_augment,
    "缩放增强": scale_augment,
    "透视增强": perspective_augment,
    "上下翻转增强": flip_vertical_augment,
    "左右翻转增强": flip_horizontal_augment,
    "BGR通道交换增强": channel_swap_augment,
    "Mosaic增强": mosaic_augment,
    "Mixup增强": mixup_augment,
    "Cutmix增强": cutmix_augment,
    "平移增强": translate_augment,
    "剪切增强": shear_augment
}

# ================== 前端 GUI ==================
class AdvancedAugmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("高级数据增强演示")
        self.root.geometry("1300x700")
        
        self.original_img = None  # 原始图片
        self.current_img = None   # 当前显示的图片
        self.img_path = None      # 图片路径
        self.other_images = []    # 用于Mosaic、Mixup、Cutmix的其他图片
        
        # 创建主框架
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 选择图片按钮
        tk.Button(control_frame, text="选择主图片", command=self.load_main_image,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)
        
        # 选择其他图片按钮（用于Mosaic、Mixup、Cutmix）
        tk.Button(control_frame, text="选择其他图片（可选）", command=self.load_other_images,
                  width=25, font=("微软雅黑", 11)).pack(pady=5)
        
        # 其他图片数量显示
        self.other_images_label = tk.Label(control_frame, text="其他图片数量：0", 
                                          font=("微软雅黑", 10))
        self.other_images_label.pack(pady=5)
        
        # 增强方法选择
        tk.Label(control_frame, text="增强方法:", font=("微软雅黑", 11, "bold")).pack(pady=10)
        
        self.augment_var = tk.StringVar()
        self.augment_var.set("选择增强方法")
        
        # 使用Combobox选择增强方法
        self.augment_combobox = ttk.Combobox(control_frame, textvariable=self.augment_var, 
                                            values=list(AUGMENTATIONS.keys()),
                                            font=("微软雅黑", 10), state="readonly",
                                            width=22)
        self.augment_combobox.pack(pady=5)
        
        # 应用增强按钮
        tk.Button(control_frame, text="应用增强", command=self.apply_augmentation,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)
        
        # 重置按钮
        tk.Button(control_frame, text="重置图片", command=self.reset_image,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)
        
        # 保存按钮
        tk.Button(control_frame, text="保存增强结果", command=self.save_result,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)
        
        # 批量增强按钮
        tk.Button(control_frame, text="批量应用所有增强", command=self.batch_augment,
                  width=25, font=("微软雅黑", 11)).pack(pady=10)
        
        # 右侧显示区域
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 标题行
        title_frame = tk.Frame(display_frame)
        title_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(title_frame, text="原始图片", font=("微软雅黑", 12, "bold")).pack(side=tk.LEFT, expand=True)
        tk.Label(title_frame, text="增强后图片", font=("微软雅黑", 12, "bold")).pack(side=tk.RIGHT, expand=True)
        
        # 图片显示行
        img_frame = tk.Frame(display_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 原始图片显示
        self.original_panel = tk.Label(img_frame, relief=tk.SUNKEN, bd=2)
        self.original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 增强后图片显示
        self.augmented_panel = tk.Label(img_frame, relief=tk.SUNKEN, bd=2)
        self.augmented_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 初始化显示
        self.init_display()
    
    def init_display(self):
        """
        初始化显示区域
        """
        # 创建空白图像
        blank_img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        blank_img = cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(blank_img)
        img_tk = ImageTk.PhotoImage(img)
        
        self.original_panel.config(image=img_tk)
        self.original_panel.image = img_tk
        
        self.augmented_panel.config(image=img_tk)
        self.augmented_panel.image = img_tk
    
    def load_main_image(self):
        """
        加载主图片
        """
        path = filedialog.askopenfilename(
            title="选择主图片",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        
        if not path:
            return
        
        print(f"尝试读取图片：{path}")
        
        # 检查文件是否存在
        if not os.path.exists(path):
            messagebox.showerror("错误", f"文件不存在：{path}")
            print(f"文件不存在：{path}")
            return
        
        # 检查文件是否为图片文件
        ext = os.path.splitext(path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            messagebox.showerror("错误", f"不支持的文件格式：{ext}\n仅支持 JPG、PNG 格式")
            print(f"不支持的文件格式：{ext}")
            return
        
        # 检查文件大小是否为0
        if os.path.getsize(path) == 0:
            messagebox.showerror("错误", f"文件为空：{path}")
            print(f"文件为空：{path}")
            return
        
        # 尝试读取图片
        try:
            img = cv2.imread(path)
            if img is None:
                # 尝试使用PIL读取
                try:
                    pil_img = Image.open(path)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    messagebox.showerror("错误", f"无法读取图片：{path}\n请检查文件完整性\n错误信息：{str(e)}")
                    print(f"无法读取图片：{path}，错误：{str(e)}")
                    return
        except Exception as e:
            messagebox.showerror("错误", f"读取图片时发生异常：{path}\n错误信息：{str(e)}")
            print(f"读取图片异常：{path}，错误：{str(e)}")
            return
        
        self.img_path = path
        self.original_img = img
        self.current_img = None
        
        # 显示原始图片
        self.display_image(self.original_img, self.original_panel)
        self.display_image(self.original_img, self.augmented_panel)
        print(f"成功读取图片：{path}")
    
    def load_other_images(self):
        """
        加载用于Mosaic、Mixup、Cutmix的其他图片
        """
        paths = filedialog.askopenfilenames(
            title="选择其他图片",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        
        if not paths:
            return
        
        print(f"尝试读取 {len(paths)} 张其他图片")
        
        # 读取图片
        new_images = []
        failed_count = 0
        
        for path in paths:
            try:
                print(f"处理图片：{path}")
                
                # 检查文件是否存在
                if not os.path.exists(path):
                    print(f"文件不存在：{path}")
                    failed_count += 1
                    continue
                
                # 检查文件大小
                if os.path.getsize(path) == 0:
                    print(f"文件为空：{path}")
                    failed_count += 1
                    continue
                
                # 尝试使用cv2读取
                img = cv2.imread(path)
                if img is None:
                    # 尝试使用PIL读取
                    try:
                        pil_img = Image.open(path)
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"无法读取图片：{path}，错误：{str(e)}")
                        failed_count += 1
                        continue
                
                new_images.append(img)
                print(f"成功读取：{path}")
            except Exception as e:
                print(f"处理图片时发生异常：{path}，错误：{str(e)}")
                failed_count += 1
                continue
        
        # 添加到其他图片列表
        self.other_images.extend(new_images)
        self.other_images_label.config(text=f"其他图片数量：{len(self.other_images)}")
        
        success_msg = f"成功加载 {len(new_images)} 张图片"
        if failed_count > 0:
            success_msg += f"，失败 {failed_count} 张"
        
        messagebox.showinfo("成功", success_msg)
        print(success_msg)
    
    def display_image(self, img, panel):
        """
        在指定面板显示图片
        """
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 获取面板大小
        panel_width = panel.winfo_width() if panel.winfo_width() > 1 else 300
        panel_height = panel.winfo_height() if panel.winfo_height() > 1 else 300
        
        # 调整图片大小以适应面板，保持宽高比
        h, w = img_rgb.shape[:2]
        scale = min(panel_width / w, panel_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        
        # 转换为PhotoImage并显示
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        panel.config(image=img_tk)
        panel.image = img_tk
    
    def apply_augmentation(self):
        """
        应用选择的数据增强
        """
        if self.original_img is None:
            messagebox.showerror("错误", "请先选择主图片")
            return
        
        aug_method = self.augment_var.get()
        if aug_method == "选择增强方法":
            messagebox.showerror("错误", "请选择增强方法")
            return
        
        try:
            # 应用增强
            if aug_method == "Mosaic增强":
                # Mosaic增强需要其他图片
                augmented_img = AUGMENTATIONS[aug_method](self.original_img, self.other_images)
            elif aug_method in ["Mixup增强", "Cutmix增强"]:
                # Mixup和Cutmix增强需要一张其他图片
                other_img = None
                if self.other_images:
                    other_img = random.choice(self.other_images)
                augmented_img = AUGMENTATIONS[aug_method](self.original_img, other_img)
            else:
                # 其他增强方法只需要主图片
                augmented_img = AUGMENTATIONS[aug_method](self.original_img.copy())
            
            self.current_img = augmented_img
            
            # 显示增强后的图片
            self.display_image(augmented_img, self.augmented_panel)
        except Exception as e:
            messagebox.showerror("错误", f"增强失败：{str(e)}")
            print(f"增强异常：{str(e)}")
    
    def reset_image(self):
        """
        重置图片
        """
        if self.original_img is None:
            return
        
        # 重置增强状态
        self.current_img = None
        
        # 显示原始图片
        self.display_image(self.original_img, self.original_panel)
        self.display_image(self.original_img, self.augmented_panel)
    
    def save_result(self):
        """
        保存增强结果
        """
        if self.original_img is None:
            messagebox.showerror("错误", "请先选择主图片")
            return
        
        if self.current_img is None:
            # 如果还没有应用增强，保存原始图片
            img_to_save = self.original_img
            default_filename = "original.jpg"
        else:
            # 保存增强后的图片
            img_to_save = self.current_img
            aug_method = self.augment_var.get()
            if aug_method == "选择增强方法":
                aug_method = "augmented"
            default_filename = f"{aug_method}.jpg"
        
        # 弹出保存对话框
        save_path = filedialog.asksaveasfilename(
            title="保存增强结果",
            defaultextension=".jpg",
            initialfile=default_filename,
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png"), ("所有文件", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 使用PIL保存图片
            img_rgb = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.save(save_path)
            messagebox.showinfo("成功", f"图片已成功保存到 {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{str(e)}")
            print(f"保存异常：{str(e)}")
    
    def batch_augment(self):
        """
        批量应用所有增强方法，并保存结果
        """
        if self.original_img is None:
            messagebox.showerror("错误", "请先选择主图片")
            return
        
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        try:
            # 批量应用所有增强方法
            success_count = 0
            for aug_name, aug_func in AUGMENTATIONS.items():
                try:
                    # 应用增强
                    if aug_name == "Mosaic增强":
                        augmented_img = aug_func(self.original_img, self.other_images)
                    elif aug_name in ["Mixup增强", "Cutmix增强"]:
                        other_img = None
                        if self.other_images:
                            other_img = random.choice(self.other_images)
                        augmented_img = aug_func(self.original_img, other_img)
                    else:
                        augmented_img = aug_func(self.original_img.copy())
                    
                    # 保存结果
                    save_path = os.path.join(save_dir, f"{aug_name}.jpg")
                    img_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    pil_img.save(save_path)
                    success_count += 1
                except Exception as e:
                    print(f"应用{aug_name}失败：{str(e)}")
            
            messagebox.showinfo("成功", f"批量增强完成，成功保存 {success_count} 张图片到 {save_dir}")
        except Exception as e:
            messagebox.showerror("错误", f"批量增强失败：{str(e)}")
            print(f"批量增强异常：{str(e)}")

# ================== 主程序 ==================
if __name__ == "__main__":
    root = tk.Tk()
    AdvancedAugmentationGUI(root)
    root.mainloop()
