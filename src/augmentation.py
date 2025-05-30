import numpy as np
import cv2


class TarTanAir_Augmentation:
    def __init__(self):
        # 相机内参矩阵（可根据你实际相机修改）
        self.fx = 320  # focal length x
        self.fy = 320  # focal length y
        self.cx = 320  # principal point x
        self.cy = 240  # principal point y
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.points3d = None

    def reproj_r_t(self, rgb, depth, R, t, inpaint=False):
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # 有效深度mask
        valid = depth > 0
        Z = depth[valid]
        X = (u[valid] - self.cx) * Z / self.fx
        Y = (v[valid] - self.cy) * Z / self.fy
        self.points3d = np.stack((X, Y, Z), axis=1)

        # 应用变换
        points3d_trans = self.points3d @ R.T + t

        # 投影回2D
        x_proj = (points3d_trans[:, 0] * self.fx) / points3d_trans[:, 2] + self.cx
        y_proj = (points3d_trans[:, 1] * self.fy) / points3d_trans[:, 2] + self.cy

        # 仅保留在图像范围内的像素
        valid_proj = (
            (x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)
        )
        x_proj = x_proj[valid_proj]
        y_proj = y_proj[valid_proj]
        src_x = u[valid][valid_proj].astype(int)
        src_y = v[valid][valid_proj].astype(int)

        # 获取颜色
        color = rgb[src_y, src_x]  # shape: (N, 3)

        # 创建空图像和Z-buffer
        aug_img = np.zeros_like(rgb)
        depth_buffer = np.zeros((height, width), dtype=np.float32)
        depth_buffer[:, :] = 1e6  # 大初值

        # 插入颜色（按最近深度）
        for idx in range(len(x_proj)):
            x = int(round(x_proj[idx]))
            y = int(round(y_proj[idx]))
            if 0 <= x < width and 0 <= y < height:
                z = points3d_trans[idx, 2]
                if z < depth_buffer[y, x]:
                    aug_img[y, x] = color[idx]
                    depth_buffer[y, x] = z

        if inpaint:
            gray_mask = (
                (aug_img[:, :, 0] == 0)
                & (aug_img[:, :, 1] == 0)
                & (aug_img[:, :, 2] == 0)
            )
            mask = gray_mask.astype(np.uint8) * 255
            aug_img = cv2.inpaint(aug_img, mask, 3, cv2.INPAINT_TELEA)

        return aug_img

    def generate_trajectory_with_gt(
        self, rgb, depth, R0, t0, delta_rotations, delta_translations, inpaint=False
    ):
        """
        输入：
          - rgb, depth：初始图像
          - R0, t0：初始绝对位姿（3x3矩阵，3维向量）
          - delta_rotations：增量旋转列表，shape (N, 3, 3)
          - delta_translations：增量平移列表，shape (N, 3)

        输出：
          - images：生成的轨迹图像序列
          - gt_poses：对应绝对位姿序列 [(R_i, t_i), ...]
        """
        images = []
        gt_poses = []

        R_curr = R0.copy()
        t_curr = t0.copy()

        for dR, dt in zip(delta_rotations, delta_translations):
            # 更新绝对位姿：左乘旋转，平移递推
            R_curr = dR @ R0
            t_curr = dR @ t0 + dt

            img = self.reproj_r_t(rgb, depth, R_curr, t_curr, inpaint=inpaint)
            images.append(img)
            gt_poses.append((R_curr.copy(), t_curr.copy()))

        return images, gt_poses


def generate_delta_rt_6directions(
    step_length=0.05, n_steps=12, rot_noise_std=0.02, trans_noise_std=0.005
):
    """
    生成沿6个主方向（左右上下前后）增量旋转和平移序列，
    平移和旋转都带小随机噪声。

    参数：
      step_length: 平移基准步长
      n_steps: 总步数
      rot_noise_std: 旋转角度噪声标准差（弧度）
      trans_noise_std: 平移噪声标准差（单位与step_length相同）

    返回：
      delta_rotations: (n_steps, 3, 3)旋转矩阵序列
      delta_translations: (n_steps, 3)平移向量序列
    """

    directions = np.array(
        [
            [1, 0, 0],  # 右
            [-1, 0, 0],  # 左
            [0, 1, 0],  # 上
            [0, -1, 0],  # 下
            [0, 0, 1],  # 前
            [0, 0, -1],  # 后
        ],
        dtype=np.float32,
    )

    delta_rotations = []
    delta_translations = []

    for i in range(n_steps):
        dir_idx = i % 6
        base_direction = directions[dir_idx]

        # 加噪声的平移 = 基础方向*step_length + 高斯噪声
        noise = np.random.normal(0, trans_noise_std, size=3)
        delta_t = base_direction * step_length + noise

        # 旋转噪声
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.normal(0, rot_noise_std)
        rvec = axis * angle
        delta_R, _ = cv2.Rodrigues(rvec)

        delta_rotations.append(delta_R)
        delta_translations.append(delta_t)

    delta_rotations = np.stack(delta_rotations, axis=0)
    delta_translations = np.stack(delta_translations, axis=0)

    return delta_rotations, delta_translations


if __name__ == "__main__":
    rgb = cv2.imread(
        "../downloaded/datasets/abandonedfactory/Hard/P000/image_left/000000_left.png"
    )
    depth = np.load(
        "../downloaded/datasets/abandonedfactory/Hard/P000/depth_left/000000_left_depth.npy"
    )  # 形状和rgb一致，单位为米
    theta = 0
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    t = np.array([-0.0, 0.0, 0.0])

    assert rgb is not None and depth is not None
    assert rgb.shape[:2] == depth.shape
    aug_tool = TarTanAir_Augmentation()
    delta_R, delta_t = generate_delta_rt_6directions(
        step_length=0.1, n_steps=16, rot_noise_std=0.01
    )
    print(delta_R.shape)  # (16, 3, 3)
    print(delta_t)  # (16, 3)
    imgs, _ = aug_tool.generate_trajectory_with_gt(rgb, depth, R, t, delta_R, delta_t, True)
    #  for img in imgs:
        #  cv2.imshow("aug_img", img)
        #  cv2.waitKey()
