import cv2
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

class DepthMapComparator:
    """
    深度图对比类
    用于比较和可视化不同算法生成的深度图
    """
    def __init__(self):
        """
        初始化深度图对比器
        """
        # 存储加载的深度图
        self.depth_maps = {}
        
        # 存储颜色映射后的深度图
        self.colored_depth_maps = {}
        
        # 深度范围
        self.min_depth = None
        self.max_depth = None
    
    def load_depth_map(self, name, file_path, is_normalized=False):
        """
        加载深度图
        
        参数:
            name: 深度图的名称标识符
            file_path: 深度图文件路径
            is_normalized: 深度图是否已归一化到[0, 1]范围
        """
        try:
            # 尝试加载深度图
            depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if depth_map is None:
                raise FileNotFoundError(f"无法读取深度图文件: {file_path}")
            
            # 如果深度图是彩色的，则转换为灰度图
            if len(depth_map.shape) == 3:
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            
            # 如果深度图是16位的，则转换为32位浮点数
            if depth_map.dtype == np.uint16:
                depth_map = depth_map.astype(np.float32)
            
            # 如果深度图没有归一化，则归一化到[0, 1]范围
            if not is_normalized:
                min_val = np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 0
                max_val = np.max(depth_map)
                if max_val > min_val:
                    depth_map = (depth_map - min_val) / (max_val - min_val)
            
            # 存储深度图
            self.depth_maps[name] = depth_map
            
            # 更新深度范围
            if self.min_depth is None or np.min(depth_map) < self.min_depth:
                self.min_depth = np.min(depth_map)
            if self.max_depth is None or np.max(depth_map) > self.max_depth:
                self.max_depth = np.max(depth_map)
            
            print(f"成功加载深度图: {name}")
            print(f"深度图尺寸: {depth_map.shape}")
            print(f"深度范围: [{np.min(depth_map):.4f}, {np.max(depth_map):.4f}]")
            
            return True
        except Exception as e:
            print(f"加载深度图 {name} 失败: {str(e)}")
            return False
    
    def normalize_depth_map(self, depth_map, min_depth=None, max_depth=None):
        """
        归一化深度图到[0, 1]范围
        
        参数:
            depth_map: 输入深度图
            min_depth: 最小深度值，如果为None则使用深度图中的最小值
            max_depth: 最大深度值，如果为None则使用深度图中的最大值
        
        返回:
            归一化后的深度图
        """
        # 复制深度图以避免修改原始数据
        normalized = depth_map.copy()
        
        # 使用提供的深度范围或计算深度图的深度范围
        if min_depth is None:
            min_depth = np.min(normalized)
        if max_depth is None:
            max_depth = np.max(normalized)
        
        # 归一化到[0, 1]范围
        if max_depth > min_depth:
            normalized = (normalized - min_depth) / (max_depth - min_depth)
            # 裁剪到[0, 1]范围
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def apply_color_map(self, name, colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None):
        """
        对深度图应用颜色映射
        
        参数:
            name: 深度图的名称标识符
            colormap: OpenCV颜色映射类型
            min_depth: 最小深度值，如果为None则使用存储的最小值
            max_depth: 最大深度值，如果为None则使用存储的最大值
        
        返回:
            颜色映射后的深度图
        """
        # 检查深度图是否已加载
        if name not in self.depth_maps:
            print(f"未找到深度图: {name}")
            return None
        
        # 获取深度图
        depth_map = self.depth_maps[name]
        
        # 归一化深度图
        normalized = self.normalize_depth_map(depth_map, min_depth, max_depth)
        
        # 转换为8位图像
        depth_8bit = np.uint8(normalized * 255)
        
        # 应用颜色映射
        colored = cv2.applyColorMap(depth_8bit, colormap)
        
        # 存储颜色映射后的深度图
        self.colored_depth_maps[name] = colored
        
        return colored
    
    def visualize_side_by_side(self, names, colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None, 
                             show_legend=True, save_path=None):
        """
        并排显示多个深度图
        
        参数:
            names: 要显示的深度图名称列表
            colormap: OpenCV颜色映射类型
            min_depth: 最小深度值，如果为None则使用存储的最小值
            max_depth: 最大深度值，如果为None则使用存储的最大值
            show_legend: 是否显示深度图例
            save_path: 保存图像的路径，如果为None则显示图像
        
        返回:
            合并后的图像
        """
        # 确保所有深度图都已加载
        for name in names:
            if name not in self.depth_maps:
                print(f"未找到深度图: {name}")
                return None
        
        # 为所有深度图应用相同的颜色映射和深度范围
        colored_maps = []
        for name in names:
            colored = self.apply_color_map(name, colormap, min_depth, max_depth)
            if colored is not None:
                colored_maps.append(colored)
        
        # 如果没有成功生成颜色映射的深度图，则返回None
        if not colored_maps:
            print("无法生成颜色映射的深度图")
            return None
        
        # 确保所有图像具有相同的高度
        heights = [cm.shape[0] for cm in colored_maps]
        width = sum(cm.shape[1] for cm in colored_maps)
        max_height = max(heights)
        
        # 创建合并后的图像
        merged = np.zeros((max_height, width, 3), dtype=np.uint8)
        
        # 放置每个深度图
        current_x = 0
        for i, (name, colored) in enumerate(zip(names, colored_maps)):
            h, w = colored.shape[:2]
            y_offset = (max_height - h) // 2
            merged[y_offset:y_offset+h, current_x:current_x+w] = colored
            
            # 添加标签
            cv2.putText(merged, name, (current_x + 10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_x += w
        
        # 添加深度图例
        if show_legend and min_depth is not None and max_depth is not None:
            # 创建图例
            legend_height = 30
            legend_width = 200
            legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
            
            # 绘制颜色条
            for x in range(legend_width):
                intensity = x / (legend_width - 1)
                color = cv2.applyColorMap(np.uint8([[intensity * 255]]), colormap)[0, 0]
                legend[:, x] = color
            
            # 添加文本标签
            cv2.putText(legend, f"{min_depth:.2f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(legend, f"{max_depth:.2f}", (legend_width - 40, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 将图例添加到合并图像的底部
            new_height = max_height + legend_height + 10
            new_merged = np.zeros((new_height, width, 3), dtype=np.uint8)
            new_merged[:max_height, :] = merged
            legend_x = (width - legend_width) // 2
            new_merged[max_height+10:new_height, legend_x:legend_x+legend_width] = legend
            merged = new_merged
        
        # 保存或显示图像
        if save_path:
            cv2.imwrite(save_path, merged)
            print(f"并排对比图像已保存到 {save_path}")
        else:
            cv2.namedWindow('Depth Maps Comparison', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth Maps Comparison', merged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return merged
    
    def compute_depth_difference(self, name1, name2, threshold=0.05, save_path=None):
        """
        计算两个深度图之间的差异
        
        参数:
            name1: 第一个深度图的名称
            name2: 第二个深度图的名称
            threshold: 差异阈值，超过此值的像素被标记为显著差异
            save_path: 保存差异图的路径，如果为None则显示图像
        
        返回:
            差异图和差异统计信息
        """
        # 检查深度图是否已加载
        if name1 not in self.depth_maps or name2 not in self.depth_maps:
            print("未找到指定的深度图")
            return None, None
        
        # 获取深度图
        depth1 = self.depth_maps[name1]
        depth2 = self.depth_maps[name2]
        
        # 确保两个深度图具有相同的尺寸
        if depth1.shape != depth2.shape:
            # 调整第二个深度图的大小以匹配第一个深度图
            depth2 = cv2.resize(depth2, (depth1.shape[1], depth1.shape[0]))
            print(f"调整 {name2} 的大小以匹配 {name1}")
        
        # 计算绝对差异
        abs_diff = np.abs(depth1 - depth2)
        
        # 计算相对差异（避免除以零）
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = abs_diff / np.maximum(depth1, depth2)
            # 处理除以零的情况
            rel_diff[np.isnan(rel_diff)] = 0
            rel_diff[np.isinf(rel_diff)] = 0
        
        # 创建差异图
        diff_map = np.zeros((depth1.shape[0], depth1.shape[1], 3), dtype=np.uint8)
        
        # 根据绝对差异和阈值创建彩色差异图
        abs_diff_normalized = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_map = cv2.applyColorMap(abs_diff_normalized, cv2.COLORMAP_JET)
        
        # 标记显著差异
        significant_diff = rel_diff > threshold
        diff_map[significant_diff] = (0, 0, 255)  # 红色标记显著差异
        
        # 计算统计信息
        valid_mask = (depth1 > 0) & (depth2 > 0)  # 只考虑有效的深度值
        if np.any(valid_mask):
            valid_abs_diff = abs_diff[valid_mask]
            valid_rel_diff = rel_diff[valid_mask]
            
            stats = {
                'name1': name1,
                'name2': name2,
                'threshold': threshold,
                'mean_absolute_error': float(np.mean(valid_abs_diff)),
                'root_mean_squared_error': float(sqrt(mean_squared_error(depth1[valid_mask], depth2[valid_mask]))),
                'mean_relative_error': float(np.mean(valid_rel_diff)),
                'median_absolute_error': float(np.median(valid_abs_diff)),
                'max_absolute_error': float(np.max(valid_abs_diff)),
                'std_absolute_error': float(np.std(valid_abs_diff)),
                'percentage_significant_diff': float(np.sum(significant_diff[valid_mask]) / np.sum(valid_mask) * 100)
            }
            
            print(f"深度图 {name1} 和 {name2} 的差异统计:")
            for key, value in stats.items():
                if key not in ['name1', 'name2', 'threshold']:
                    print(f"  {key}: {value}")
        else:
            print("没有有效的深度值进行比较")
            stats = None
        
        # 添加标签
        cv2.putText(diff_map, f"Difference: {name1} vs {name2}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存或显示差异图
        if save_path:
            cv2.imwrite(save_path, diff_map)
            print(f"差异图已保存到 {save_path}")
        else:
            cv2.namedWindow('Depth Map Difference', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth Map Difference', diff_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return diff_map, stats
    
    def create_comparison_grid(self, names, rows=2, cols=2, colormap=cv2.COLORMAP_JET, 
                             min_depth=None, max_depth=None, save_path=None):
        """
        创建深度图对比网格
        
        参数:
            names: 要显示的深度图名称列表
            rows: 网格行数
            cols: 网格列数
            colormap: OpenCV颜色映射类型
            min_depth: 最小深度值，如果为None则使用存储的最小值
            max_depth: 最大深度值，如果为None则使用存储的最大值
            save_path: 保存网格图像的路径，如果为None则显示图像
        
        返回:
            网格图像
        """
        # 确保所有深度图都已加载
        valid_names = []
        for name in names:
            if name in self.depth_maps:
                valid_names.append(name)
            else:
                print(f"未找到深度图: {name}")
        
        if not valid_names:
            print("没有有效的深度图进行对比")
            return None
        
        # 计算网格尺寸
        num_maps = min(len(valid_names), rows * cols)
        
        # 为所有深度图应用相同的颜色映射和深度范围
        colored_maps = []
        for name in valid_names[:num_maps]:
            colored = self.apply_color_map(name, colormap, min_depth, max_depth)
            if colored is not None:
                colored_maps.append((name, colored))
        
        # 计算每个单元格的大小
        cell_heights = [cm.shape[0] for _, cm in colored_maps]
        cell_widths = [cm.shape[1] for _, cm in colored_maps]
        cell_height = max(cell_heights) if cell_heights else 480
        cell_width = max(cell_widths) if cell_widths else 640
        
        # 创建网格图像
        grid_height = rows * cell_height
        grid_width = cols * cell_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # 放置每个深度图到网格中
        for i, (name, colored) in enumerate(colored_maps):
            row = i // cols
            col = i % cols
            h, w = colored.shape[:2]
            y_offset = row * cell_height + (cell_height - h) // 2
            x_offset = col * cell_width + (cell_width - w) // 2
            grid[y_offset:y_offset+h, x_offset:x_offset+w] = colored
            
            # 添加标签
            cv2.putText(grid, name, (x_offset + 10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存或显示网格图像
        if save_path:
            cv2.imwrite(save_path, grid)
            print(f"深度图对比网格已保存到 {save_path}")
        else:
            cv2.namedWindow('Depth Maps Grid', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth Maps Grid', grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return grid
    
    def save_comparison_results(self, stats, save_path='comparison_results.json'):
        """
        保存对比结果到JSON文件
        
        参数:
            stats: 对比统计信息
            save_path: 保存的文件路径
        """
        try:
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=4)
            
            print(f"对比结果已保存到 {save_path}")
            return True
        except Exception as e:
            print(f"保存对比结果失败: {str(e)}")
            return False

# 主程序
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='深度图对比工具')
    parser.add_argument('--depth_maps', type=str, nargs='+', help='深度图文件路径列表')
    parser.add_argument('--names', type=str, nargs='+', help='深度图名称列表，与文件路径一一对应')
    parser.add_argument('--normalized', action='store_true', help='深度图是否已归一化')
    parser.add_argument('--output_dir', type=str, default='.', help='结果输出目录')
    parser.add_argument('--colormap', type=str, default='jet', help='颜色映射类型')
    parser.add_argument('--min_depth', type=float, help='最小深度值')
    parser.add_argument('--max_depth', type=float, help='最大深度值')
    parser.add_argument('--show_side_by_side', action='store_true', help='并排显示深度图')
    parser.add_argument('--show_grid', action='store_true', help='以网格形式显示深度图')
    parser.add_argument('--compute_difference', nargs=2, help='计算两个深度图之间的差异')
    parser.add_argument('--diff_threshold', type=float, default=0.05, help='差异阈值')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建深度图对比实例
    comparator = DepthMapComparator()
    
    # 加载深度图
    if args.depth_maps:
        # 如果没有提供名称，则使用文件名作为名称
        if not args.names or len(args.names) != len(args.depth_maps):
            args.names = [os.path.splitext(os.path.basename(path))[0] for path in args.depth_maps]
        
        for name, path in zip(args.names, args.depth_maps):
            comparator.load_depth_map(name, path, args.normalized)
    
    # 获取颜色映射
    colormap_dict = {
        'autumn': cv2.COLORMAP_AUTUMN,
        'bone': cv2.COLORMAP_BONE,
        'jet': cv2.COLORMAP_JET,
        'winter': cv2.COLORMAP_WINTER,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'ocean': cv2.COLORMAP_OCEAN,
        'summer': cv2.COLORMAP_SUMMER,
        'spring': cv2.COLORMAP_SPRING,
        'cool': cv2.COLORMAP_COOL,
        'hsv': cv2.COLORMAP_HSV,
        'pink': cv2.COLORMAP_PINK,
        'hot': cv2.COLORMAP_HOT
    }
    
    colormap = colormap_dict.get(args.colormap.lower(), cv2.COLORMAP_JET)
    
    # 并排显示深度图
    if args.show_side_by_side and args.names:
        save_path = os.path.join(args.output_dir, 'side_by_side_comparison.png')
        comparator.visualize_side_by_side(
            args.names, 
            colormap=colormap,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            save_path=save_path
        )
    
    # 以网格形式显示深度图
    if args.show_grid and args.names:
        # 计算网格行数和列数
        num_maps = len(args.names)
        cols = int(sqrt(num_maps)) + 1
        rows = (num_maps + cols - 1) // cols
        
        save_path = os.path.join(args.output_dir, 'grid_comparison.png')
        comparator.create_comparison_grid(
            args.names,
            rows=rows,
            cols=cols,
            colormap=colormap,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            save_path=save_path
        )
    
    # 计算两个深度图之间的差异
    if args.compute_difference and len(args.compute_difference) == 2:
        name1, name2 = args.compute_difference
        save_path = os.path.join(args.output_dir, f'difference_{name1}_vs_{name2}.png')
        diff_map, stats = comparator.compute_depth_difference(
            name1, 
            name2,
            threshold=args.diff_threshold,
            save_path=save_path
        )
        
        # 保存对比结果
        if stats:
            results_path = os.path.join(args.output_dir, f'comparison_stats_{name1}_vs_{name2}.json')
            comparator.save_comparison_results(stats, results_path)