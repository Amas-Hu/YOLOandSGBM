# -*- coding: utf-8 -*-
"""
StarBoost-V3 UCT-SGM + 外部背景融合
1) CLAHE + 双边滤波预处理，增强水下低对比度场景；
2) AD-Census 代价构造，提高低纹理区域匹配鲁棒性；
3) 纹理阈值生成 ROI，只保留手/海星/墙壁等结构区域的视差信息；
4) speckle 连通域去噪 + 形态学处理 + 连通域填充，补满手臂和海星空洞；
5) 墙壁等结构边缘保留原始 SGM 视差；
6) 自定义黄–蓝渐变配色；
7) 使用 ROI 将本方法的前景贴到另一张视差图的背景上。
"""

import cv2
import numpy as np
import time


# ==================== 相机参数 ====================
class CameraConfigs:
    def __init__(self):
        # 图像尺寸（按你的标定来）
        self.size = (1920, 1080)

        left_camera_matrix = np.array([
            [1.481512980952683e+03, 0, 9.741930028279249e+02],
            [0, 1.480735907711620e+03, 5.435121362485065e+02],
            [0, 0, 1]
        ])
        left_distortion = np.array(
            [[0.302369610254450, -0.832843922384300,
              0.001560565758676, -0.005541012945384,
              6.575484609862809]]
        )

        right_camera_matrix = np.array([
            [1.482296723453751e+03, 0, 9.660834185159497e+02],
            [0, 1.482629468412100e+03, 5.492517774177843e+02],
            [0, 0, 1]
        ])
        right_distortion = np.array(
            [[0.294255745818360, 0.033358045248374,
              -0.001227604040424, 0.002781368162044,
              1.366075774860986]]
        )

        R = np.array(
            [[0.999907272711480, -9.336217474162630e-04, -0.013585813524483],
             [9.689518153560432e-04, 0.999996165720932, 0.002594161870988],
             [0.013583339466744, -0.002607085320069, 0.999904343422442]]
        )
        T = np.array(
            [-1.202021696951524e+02, -0.010096027651827, -0.512882222667619]
        )

        self.left_camera_matrix = left_camera_matrix.astype(np.float64)
        self.left_distortion = left_distortion.astype(np.float64)
        self.right_camera_matrix = right_camera_matrix.astype(np.float64)
        self.right_distortion = right_distortion.astype(np.float64)
        self.R = R.astype(np.float64)
        self.T = T.reshape(3, 1).astype(np.float64)

        # 立体校正
        self.R1, self.R2, self.P1, self.P2, self.Q, \
            self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
                self.left_camera_matrix, self.left_distortion,
                self.right_camera_matrix, self.right_distortion,
                self.size, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0
            )

        # 预计算重映射表
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion,
            self.R1, self.P1, self.size, cv2.CV_16SC2
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion,
            self.R2, self.P2, self.size, cv2.CV_16SC2
        )


def rectify_pair(cfg: CameraConfigs, imgL, imgR):
    """将左右原图矫正到共面、共极线坐标系"""
    if imgL.ndim == 3:
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    else:
        grayL = imgL.copy()
    if imgR.ndim == 3:
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    else:
        grayR = imgR.copy()
    rectL = cv2.remap(grayL, cfg.left_map1, cfg.left_map2, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, cfg.right_map1, cfg.right_map2, cv2.INTER_LINEAR)
    return rectL, rectR


# ==================== StarBoost-V3 UCT-SGM ====================
class StarBoostUCTSGM_V3:
    def __init__(self,
                 camera_configs: CameraConfigs,
                 num_disparities=64,
                 census_ksize=5):
        self.cfg = camera_configs
        self.num_disparities = num_disparities
        self.census_ksize = census_ksize

        # 汉明距离查找表
        self._popcount_lut = np.array(
            [bin(i).count("1") for i in range(256)], dtype=np.uint8
        )

    # -------- 预处理：CLAHE + 双边滤波 --------
    def _preprocess(self, gray):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g1 = clahe.apply(gray)
        g2 = cv2.bilateralFilter(g1, d=5, sigmaColor=50, sigmaSpace=50)
        return g2

    # -------- Census 变换 --------
    def _census(self, gray, ksize=5):
        h, w = gray.shape
        r = ksize // 2
        census = np.zeros((h, w), dtype=np.uint32)
        center = gray[r:h - r, r:w - r]
        bit = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                patch = gray[r + dy:h - r + dy, r + dx:w - r + dx]
                census[r:h - r, r:w - r] |= (
                    (patch < center).astype(np.uint32) << bit
                )
                bit += 1
        return census

    # -------- AD-Census 代价体构造 --------
    def _build_cost_volume(self, left, right, max_disp):
        h, w = left.shape
        cost_vol = np.full((h, w, max_disp), 1e6, dtype=np.float32)

        lf = left.astype(np.float32)
        rf = right.astype(np.float32)
        cL = self._census(left, self.census_ksize)
        cR = self._census(right, self.census_ksize)

        for d in range(max_disp):
            xL0, xL1 = 0, w
            xR0, xR1 = xL0 - d, xL1 - d
            if xR1 <= 0:
                continue
            xL0v = max(xL0, d)
            xR0v = xL0v - d
            width = min(xL1, w) - xL0v
            if width <= 0:
                continue

            IL = lf[:, xL0v:xL0v + width]
            IR = rf[:, xR0v:xR0v + width]
            ad = np.abs(IL - IR)
            ad_norm = ad / (ad.max() + 1e-6)

            CL = cL[:, xL0v:xL0v + width]
            CR = cR[:, xR0v:xR0v + width]
            xor = np.bitwise_xor(CL, CR).view(np.uint8)
            xor = xor.reshape(h, width, 4)
            ham = self._popcount_lut[xor].sum(axis=2).astype(np.float32)
            ham_norm = ham / (ham.max() + 1e-6)

            alpha = 0.7
            cost = alpha * ad_norm + (1.0 - alpha) * ham_norm
            cost_vol[:, xL0v:xL0v + width, d] = cost

        return cost_vol

    # -------- 纹理估计：梯度幅值 --------
    def _texture(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return mag / (mag.max() + 1e-6)

    # -------- 纹理引导 SGM 聚合 (4 方向 UCT-SGM) --------
    def _sgm_aggregate(self, cost_vol, tex,
                        P1_base=1.5, P2_base=12.0):
        H, W, D = cost_vol.shape
        big = 1e9

        def aggregate_direction(cost, tex, axis, forward=True):
            L = np.full_like(cost, big, dtype=np.float32)
            if axis == 1:  # 水平路径
                if forward:
                    L[:, 0, :] = cost[:, 0, :]
                    rng = range(1, W)
                else:
                    L[:, W - 1, :] = cost[:, W - 1, :]
                    rng = range(W - 2, -1, -1)
                for x in rng:
                    xp = x - 1 if forward else x + 1
                    prev = L[:, xp, :]
                    C = cost[:, x, :]
                    t = tex[:, xp]
                    P1 = P1_base / (1.0 + t)
                    P2 = P2_base / (1.0 + t)
                    P1r = P1[:, None]
                    P2r = P2[:, None]
                    min_prev = prev.min(axis=1, keepdims=True)

                    prev_d = prev
                    prev_dm1 = np.empty_like(prev)
                    prev_dp1 = np.empty_like(prev)
                    prev_dm1[:, 0] = big
                    prev_dm1[:, 1:] = prev[:, :-1]
                    prev_dp1[:, -1] = big
                    prev_dp1[:, :-1] = prev[:, 1:]

                    c3 = np.broadcast_to(min_prev + P2r, prev.shape)
                    c0 = prev_d
                    c1 = prev_dm1 + P1r
                    c2 = prev_dp1 + P1r
                    t_val = np.minimum(
                        np.minimum(c0, c1),
                        np.minimum(c2, c3)
                    )
                    L[:, x, :] = C + t_val - min_prev
            else:          # 垂直路径
                if forward:
                    L[0, :, :] = cost[0, :, :]
                    rng = range(1, H)
                else:
                    L[H - 1, :, :] = cost[H - 1, :, :]
                    rng = range(H - 2, -1, -1)
                for y in rng:
                    yp = y - 1 if forward else y + 1
                    prev = L[yp, :, :]
                    C = cost[y, :, :]
                    t = tex[yp, :]
                    P1 = P1_base / (1.0 + t)
                    P2 = P2_base / (1.0 + t)
                    P1r = P1[:, None]
                    P2r = P2[:, None]
                    min_prev = prev.min(axis=1, keepdims=True)

                    prev_d = prev
                    prev_dm1 = np.empty_like(prev)
                    prev_dp1 = np.empty_like(prev)
                    prev_dm1[:, 0] = big
                    prev_dm1[:, 1:] = prev[:, :-1]
                    prev_dp1[:, -1] = big
                    prev_dp1[:, :-1] = prev[:, 1:]

                    c3 = np.broadcast_to(min_prev + P2r, prev.shape)
                    c0 = prev_d
                    c1 = prev_dm1 + P1r
                    c2 = prev_dp1 + P1r
                    t_val = np.minimum(
                        np.minimum(c0, c1),
                        np.minimum(c2, c3)
                    )
                    L[y, :, :] = C + t_val - min_prev
            return L

        L_lr = aggregate_direction(cost_vol, tex, axis=1, forward=True)
        L_rl = aggregate_direction(cost_vol, tex, axis=1, forward=False)
        L_tb = aggregate_direction(cost_vol, tex, axis=0, forward=True)
        L_bt = aggregate_direction(cost_vol, tex, axis=0, forward=False)
        return (L_lr + L_rl + L_tb + L_bt) / 4.0

    # -------- speckle 去噪 + 前景填充 + 墙壁轮廓保留 --------
    def _post_process_disp(self, disp, tex):
        """
        disp : H×W 视差索引（argmin 输出）
        tex  : H×W 纹理强度 (0~1)
        返回: 处理后的视差 disp_f, 实心前景 ROI 掩膜 roi_filled
        """
        # 0. 基础处理
        disp_f = disp.astype(np.float32)
        disp_f[disp_f < 0] = 0
        disp_orig = disp_f.copy()   # 保存原始 SGM 视差，用于墙壁等结构

        H, W = disp_f.shape

        # 1) speckle 去噪（连通域）
        tmp = (disp_f * 16).astype(np.int16)
        cv2.filterSpeckles(tmp, 0, 300, 8)
        disp_f = tmp.astype(np.float32) / 16.0
        disp_f[disp_f < 0] = 0

        # 2) 纹理图 → 边缘 ROI（墙壁斜线 + 手 + 海星）
        tex_norm = tex / (tex.max() + 1e-6)
        edge_roi = (tex_norm > 0.12).astype(np.uint8)   # 阈值稍低，保证手臂完整

        # 3) 形态学 + 连通域面积过滤
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edge_clean = cv2.morphologyEx(edge_roi, cv2.MORPH_OPEN, k3, iterations=1)
        edge_clean = cv2.dilate(edge_clean, k7, iterations=2)
        edge_clean = cv2.morphologyEx(edge_clean, cv2.MORPH_CLOSE, k7, iterations=2)

        num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            edge_clean, connectivity=8
        )
        min_area_ratio = 0.001   # 可调：0.0008 ~ 0.002
        min_area = int(min_area_ratio * H * W)

        edge_filtered = np.zeros_like(edge_clean, dtype=np.uint8)
        for lab in range(1, num_lbl):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_area:
                edge_filtered[labels == lab] = 1

        edge_clean = edge_filtered
        edge_u8 = (edge_clean * 255).astype(np.uint8)

        # 4) 轮廓填充 → 实心前景 ROI（手 + 海星整体）
        contours, _ = cv2.findContours(edge_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        roi_filled = np.zeros_like(edge_u8)
        if len(contours) > 0:
            cv2.drawContours(roi_filled, contours, -1, 255, thickness=-1)
        roi_filled = (roi_filled > 0).astype(np.uint8)

        # 轻微膨胀+收缩，让手臂/海星边缘更饱满
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_filled = cv2.dilate(roi_filled, k5, iterations=1)
        roi_filled = cv2.erode(roi_filled, k3, iterations=1)

        if roi_filled.sum() < 0.005 * H * W:
            print("[StarBoost] ROI too small, fallback to full image.")
            roi_filled[:] = 1

        # 拆分：前景主体 / 结构边缘 / 背景
        fg_mask   = roi_filled.astype(bool)               # 手 + 海星
        edge_mask = edge_clean.astype(bool) & ~fg_mask    # 墙壁边缘
        bg_mask   = ~(fg_mask | edge_mask)                # 纯水体

        # 5) 背景置零
        disp_f[bg_mask] = 0

        # 6) 前景连通域填充
        num_labels, labels = cv2.connectedComponents(fg_mask.astype(np.uint8))
        for lab in range(1, num_labels):
            mask = (labels == lab)
            valid = mask & (disp_f > 0)
            if np.any(valid):
                median_val = float(np.median(disp_f[valid]))
                disp_f[mask] = median_val

        # 7) 结构边缘（墙壁等）保留原始 SGM 视差，只在边缘区写回
        if edge_mask.any():
            edge_disp = disp_orig.copy()
            edge_disp[~edge_mask] = 0
            edge_u8 = cv2.normalize(edge_disp, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            edge_u8 = cv2.medianBlur(edge_u8, 3)
            edge_u8 = cv2.bilateralFilter(edge_u8, d=5,
                                          sigmaColor=50, sigmaSpace=50)
            edge_disp = edge_u8.astype(np.float32)
            disp_f[edge_mask] = edge_disp[edge_mask]

        # 8) 全局轻微平滑
        disp_u8 = cv2.normalize(disp_f, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
        disp_u8 = cv2.medianBlur(disp_u8, 3)
        disp_u8 = cv2.bilateralFilter(disp_u8, d=5,
                                      sigmaColor=50, sigmaSpace=50)
        disp_f = disp_u8.astype(np.float32)

        return disp_f, roi_filled

    # -------- 主流程：一定要在类里面，注意缩进 --------
    def compute_disparity(self, rectL, rectR):
        h, w = rectL.shape

        left = self._preprocess(rectL)
        right = self._preprocess(rectR)

        tex = self._texture(left)

        print(f"[StarBoost-V3] build cost volume: H={h}, W={w}, D={self.num_disparities}")
        cost_vol = self._build_cost_volume(left, right, self.num_disparities)

        print("[StarBoost-V3] texture-guided SGM aggregation ...")
        agg_cost = self._sgm_aggregate(cost_vol, tex)

        disp = np.argmin(agg_cost, axis=2).astype(np.float32)

        disp_pp, roi = self._post_process_disp(disp, tex)
        return disp_pp, roi

    # -------- 自定义黄–蓝渐变配色 --------
    @staticmethod
    def disp_to_color(disp, roi=None):
        """
        将视差映射为颜色：
        - 背景：深蓝色
        - 前景（手臂 + 海星）：从蓝色逐渐过渡到黄色（视差越大越偏黄）
        """
        H, W = disp.shape
        mask = disp > 0
        if mask.sum() > 0:
            vmin = disp[mask].min()
            vmax = disp[mask].max()
        else:
            vmin, vmax = 0, 1

        disp_norm = (disp - vmin) / (vmax - vmin + 1e-6)
        disp_norm = np.clip(disp_norm, 0.0, 1.0).astype(np.float32)

        # 背景初始化为深蓝
        color = np.zeros((H, W, 3), dtype=np.uint8)
        color[:] = (128, 0, 0)  # BGR: 深蓝

        # 前景区域：用 HSV 做黄–蓝渐变
        if roi is not None:
            fg = (roi > 0) & mask
        else:
            fg = mask

        if np.any(fg):
            v = disp_norm[fg]  # 0~1

            # Hue: 从蓝色(120) → 黄色(30)，视差越大颜色越暖
            H_ch = 120.0 - 90.0 * v
            H_ch = np.clip(H_ch, 30.0, 120.0)

            S_ch = 255.0 * np.ones_like(v)
            V_ch = 255.0 * np.ones_like(v)

            hsv = np.zeros((v.shape[0], 1, 3), dtype=np.uint8)
            hsv[:, 0, 0] = H_ch.astype(np.uint8)
            hsv[:, 0, 1] = S_ch.astype(np.uint8)
            hsv[:, 0, 2] = V_ch.astype(np.uint8)

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]

            ys, xs = np.where(fg)
            color[ys, xs, :] = bgr

        return color


# ==================== 运行 & 与外部背景融合 ====================
def run_starboost_v3(left_path, right_path,
                     out_disp="disp_StarBoostV3_blend.png",
                     out_roi="roi_StarBoostV3.png",
                     bg_path=None):
    """
    bg_path: 第二张视差图（你想要的背景）的路径。
             如果为 None，则只输出本方法的 disp_color。
    """
    cfg = CameraConfigs()
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    if imgL is None or imgR is None:
        print("无法读取左右图像，请检查路径")
        return

    rectL, rectR = rectify_pair(cfg, imgL, imgR)

    matcher = StarBoostUCTSGM_V3(cfg, num_disparities=64, census_ksize=5)

    t0 = time.time()
    disp, roi = matcher.compute_disparity(rectL, rectR)
    t1 = time.time()
    print(f"[StarBoost-V3] 用时 {t1 - t0:.2f} s")

    # 第一张：StarBoost 的彩色视差（前景颜色）
    disp_color = matcher.disp_to_color(disp, roi=roi)

    # 如果给了第二张背景视差图，就做前景/背景融合
    if bg_path is not None:
        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            print("无法读取背景图像 bg_path，改为仅使用 disp_color")
            final_color = disp_color
        else:
            # 调整背景大小与 disp_color 一致
            H, W, _ = disp_color.shape
            bg_img = cv2.resize(bg_img, (W, H), interpolation=cv2.INTER_LINEAR)

            # roi 为前景掩膜：前景用 disp_color，背景用 bg_img
            mask_fg = (roi > 0).astype(np.uint8)
            mask_fg_3 = np.repeat(mask_fg[:, :, None], 3, axis=2)

            final_color = bg_img.copy()
            final_color[mask_fg_3 == 1] = disp_color[mask_fg_3 == 1]
    else:
        final_color = disp_color

    cv2.imwrite(out_disp, final_color)
    cv2.imwrite(out_roi, (roi * 255).astype(np.uint8))
    print(f"[StarBoost-V3] 融合视差图已保存: {out_disp}")
    print(f"[StarBoost-V3] ROI 掩膜已保存: {out_roi}")

    cv2.imshow("Left", imgL)
    cv2.imshow("Disp StarBoost-V3 (blended)", final_color)
    cv2.imshow("ROI (texture mask)", roi * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 你的左右图（保持不变）
    left_image_path  = r"..\0_image\100left\left_13.jpg"
    right_image_path = r"..\0_image\100right\right_13.jpg"

    # 第二张视差图（你想要的背景）——这里改成你的那张 PNG 的路径
    # 例如：r"..\0_image\disp_bg_sgbm.png"
    bg_disp_path = r"图片1.png"

    run_starboost_v3(left_image_path,
                     right_image_path,
                     out_disp="disp_StarBoostV3_blend.png",
                     out_roi="roi_StarBoostV3.png",
                     bg_path=bg_disp_path)
