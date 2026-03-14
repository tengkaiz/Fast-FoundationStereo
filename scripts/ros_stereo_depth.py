#!/usr/bin/env python3
"""
ROS 节点：订阅双目图像，实时推理深度并发布深度图和点云。

支持两种后端：
  - pytorch
  - tensorrt
"""

import argparse
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from core.foundation_stereo import TrtRunner
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, depth2xyzmap, set_logging_format, set_seed, toOpen3dCloud, o3d


rospy = None
Image = None
CameraInfo = None
PointCloud2 = None
PointField = None
Header = None
CvBridge = None
tf2_ros = None
tf_transformations = None


def ensure_ros_imported():
    global rospy, Image, CameraInfo, PointCloud2, PointField, Header, CvBridge
    global tf2_ros, tf_transformations
    if rospy is not None:
        return

    try:
        import rospy as _rospy
        from cv_bridge import CvBridge as _CvBridge
        from sensor_msgs.msg import CameraInfo as _CameraInfo
        from sensor_msgs.msg import Image as _Image
        from sensor_msgs.msg import PointCloud2 as _PointCloud2
        from sensor_msgs.msg import PointField as _PointField
        from std_msgs.msg import Header as _Header
        import tf2_ros as _tf2_ros
        import tf.transformations as _tf_transformations
    except ImportError as exc:
        raise ImportError("需要 ROS 环境，请先 source /opt/ros/<distro>/setup.bash") from exc

    rospy = _rospy
    Image = _Image
    CameraInfo = _CameraInfo
    PointCloud2 = _PointCloud2
    PointField = _PointField
    Header = _Header
    CvBridge = _CvBridge
    tf2_ros = _tf2_ros
    tf_transformations = _tf_transformations


def build_point_cloud2(header, points, colors):
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    r = colors[:, 0].astype(np.uint32)
    g = colors[:, 1].astype(np.uint32)
    b = colors[:, 2].astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    buf = np.empty(
        len(points),
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)],
    )
    buf["x"] = points[:, 0]
    buf["y"] = points[:, 1]
    buf["z"] = points[:, 2]
    buf["rgb"] = rgb_packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points)
    msg.data = buf.tobytes()
    msg.is_dense = False
    return msg


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_cli_value(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return list(value)
    return value


def build_parser():
    default_model_dir = os.path.join(code_dir, "..", "weights/23-36-37/model_best_bp2_serialize.pth")
    default_onnx_dir = os.path.join(code_dir, "..", "output/23-36-37")
    parser = argparse.ArgumentParser(
        description="ROS 实时双目深度推理节点",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--backend", type=str, choices=["pytorch", "tensorrt"], default="tensorrt", help="选择推理后端")

    parser.add_argument("--model_dir", type=str, default=default_model_dir)
    parser.add_argument("--onnx_dir", type=str, default=default_onnx_dir)

    # 推理参数
    parser.add_argument("--valid_iters", type=int, default=8, help="GRU 迭代次数，越少越快")
    parser.add_argument("--max_disp", type=int, default=192, help="最大视差(像素)")
    parser.add_argument("--scale", type=float, default=1.0, help="PyTorch 后端图像缩放因子")
    parser.add_argument("--hiera", type=int, default=0, help="层级推理: 0=标准, 1=二阶段(大场景更好)")
    parser.add_argument("--low_memory", type=int, default=0, help="低显存模式: 0=关, 1=开")
    parser.add_argument("--optimize_build_volume", type=str, default="pytorch1",
                        choices=["pytorch1", "triton"], help="成本体积构建方式")

    # 深度/点云后处理参数
    parser.add_argument("--zfar", type=float, default=10.0, help="最大深度截断(米)")
    parser.add_argument("--zmin", type=float, default=0.1, help="最小有效深度(米)")
    parser.add_argument("--remove_invisible", type=int, default=1, help="移除右视图不可见区域: 0=否, 1=是")
    parser.add_argument("--denoise_cloud", type=int, default=1, help="深度去噪: 0=关, 1=开")
    parser.add_argument("--denoise_kernel", type=int, default=3, help="去噪: 孤立点检测窗口大小(奇数, 越大越激进)")
    parser.add_argument("--edge_filter", type=int, default=1, help="边缘飞点过滤: 0=关, 1=开")
    parser.add_argument("--edge_threshold", type=float, default=0.05, help="边缘检测阈值: 深度梯度与深度值的比值，越小越激进")
    parser.add_argument("--depth_format", type=str, default="16uc1_mm",
                        choices=["32fc1_m", "16uc1_mm"],
                        help="深度图输出格式: 16uc1_mm=uint16毫米(默认,兼容Orbbec/RealSense), 32fc1_m=float32米")

    # TensorRT 专用参数
    parser.add_argument("--cv_group", type=int, default=8, help="TensorRT cost volume group 数")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[480, 640],
        help="TensorRT engine 对应的输入尺寸",
    )

    # 深度配准：将深度从 left IR 坐标系对齐到 RGB 坐标系
    parser.add_argument("--depth_registration", type=int, default=1,
                        help="深度配准到RGB相机: 0=关(输出left_ir坐标系), 1=开(输出color坐标系)")
    parser.add_argument("--color_info_topic", type=str, default="/camera/color/camera_info")

    parser.add_argument("--frame_id", type=str, default="camera_left_ir_optical_frame")
    parser.add_argument("--left_topic", type=str, default="/camera/left_ir/image_raw")
    parser.add_argument("--right_topic", type=str, default="/camera/right_ir/image_raw")
    parser.add_argument("--left_info_topic", type=str, default="/camera/left_ir/camera_info")
    parser.add_argument("--right_info_topic", type=str, default="/camera/right_ir/camera_info")

    return parser


def merge_config(cli_args):
    """三层合并：parser 默认值 → yaml 配置 → CLI 显式传参。"""
    parser = build_parser()
    parser_defaults = vars(parser.parse_args([]))

    # 找出用户在命令行显式传入的参数（值 != parser 默认值的）
    cli_overrides = {}
    for key, value in vars(cli_args).items():
        nv = normalize_cli_value(value)
        nd = normalize_cli_value(parser_defaults.get(key))
        if nv is not None and nv != nd:
            cli_overrides[key] = nv

    # 第一层：parser 默认值
    merged = dict(parser_defaults)
    backend = cli_overrides.get("backend", merged["backend"])
    merged["backend"] = backend

    # 第二层：从 yaml 读取模型配置（可覆盖 max_disp、valid_iters 等）
    if backend == "pytorch":
        model_dir = cli_overrides.get("model_dir", merged["model_dir"])
        cfg_path = os.path.join(os.path.dirname(model_dir), "cfg.yaml")
        if os.path.isfile(cfg_path):
            merged.update(read_yaml(cfg_path))
    else:
        onnx_dir = cli_overrides.get("onnx_dir", merged["onnx_dir"])
        cfg_path = os.path.join(onnx_dir, "onnx.yaml")
        if os.path.isfile(cfg_path):
            merged.update(read_yaml(cfg_path))

    # 第三层：CLI 显式传参优先级最高
    merged.update(cli_overrides)

    # TensorRT 自动拼接 engine 路径
    if backend == "tensorrt":
        onnx_dir = merged["onnx_dir"]
        merged.setdefault("feature_engine_path", os.path.join(onnx_dir, "feature_runner.engine"))
        merged.setdefault("post_engine_path", os.path.join(onnx_dir, "post_runner.engine"))

    return OmegaConf.create(merged)


def ensure_file_exists(path, description):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{description} 不存在: {path}")


def to_rgb_image(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image[..., :3]


class PytorchBackendRunner:
    def __init__(self, args):
        ensure_file_exists(args.model_dir, "模型文件")
        self.args = args
        self.model = torch.load(args.model_dir, map_location="cpu", weights_only=False)
        self.model.args.valid_iters = args.valid_iters
        self.model.args.max_disp = args.max_disp
        self.model.cuda().eval()
        self._padder = None
        self._cached_shape = None

        rospy.loginfo(
            f"使用 PyTorch 后端 | model={args.model_dir} | "
            f"valid_iters={args.valid_iters} | max_disp={args.max_disp} | scale={args.scale} | "
            f"hiera={args.hiera} | low_memory={args.low_memory} | "
            f"optimize_build_volume={args.optimize_build_volume}"
        )
        self._warmup()

    def _warmup(self):
        rospy.loginfo("PyTorch 模型加载完成，开始预热...")
        dummy = torch.zeros(1, 3, 480, 640, device="cuda")
        padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
        d1, d2 = padder.pad(dummy, dummy)
        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            self.model.forward(d1, d2, iters=self.args.valid_iters, test_mode=True)
        torch.cuda.synchronize()
        rospy.loginfo("PyTorch 预热完成")

    def _get_padder(self, height, width):
        if self._cached_shape != (height, width):
            self._padder = InputPadder((1, 3, height, width), divis_by=32, force_square=False)
            self._cached_shape = (height, width)
        return self._padder

    def prepare_images(self, left_img, right_img):
        scale = float(self.args.scale)
        if scale != 1.0:
            left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
            right_img = cv2.resize(right_img, dsize=(left_img.shape[1], left_img.shape[0]))
            return left_img, right_img, scale
        return left_img, right_img, None

    def infer(self, left_img, right_img):
        height, width = left_img.shape[:2]
        img0 = torch.as_tensor(left_img, device="cuda").float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_img, device="cuda").float()[None].permute(0, 3, 1, 2)

        padder = self._get_padder(height, width)
        img0_pad, img1_pad = padder.pad(img0, img1)

        low_memory = bool(self.args.get("low_memory", 0))

        with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
            if self.args.get("hiera", 0):
                disp = self.model.run_hierachical(
                    img0_pad, img1_pad,
                    iters=self.args.valid_iters,
                    test_mode=True,
                    low_memory=low_memory,
                    small_ratio=0.5,
                )
            else:
                disp = self.model.forward(
                    img0_pad, img1_pad,
                    iters=self.args.valid_iters,
                    test_mode=True,
                    low_memory=low_memory,
                    optimize_build_volume=self.args.get("optimize_build_volume", "pytorch1"),
                )
        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(height, width).clip(0, None)


class TensorRTBackendRunner:
    def __init__(self, args):
        self.args = args
        ensure_file_exists(args.feature_engine_path, "TensorRT feature engine")
        ensure_file_exists(args.post_engine_path, "TensorRT post engine")
        if not args.image_size or len(args.image_size) != 2:
            raise ValueError("TensorRT 后端需要合法的 image_size=[H, W]")

        self.image_size = tuple(int(v) for v in args.image_size)
        self.model = TrtRunner(args, args.feature_engine_path, args.post_engine_path)

        rospy.loginfo(
            f"使用 TensorRT 后端 | onnx_dir={args.onnx_dir} | "
            f"image_size={self.image_size} | max_disp={args.max_disp} | cv_group={args.cv_group}"
        )

    def prepare_images(self, left_img, right_img):
        src_height, src_width = left_img.shape[:2]
        dst_height, dst_width = self.image_size
        if src_height != dst_height or src_width != dst_width:
            rospy.logwarn_once(
                f"TensorRT 输入将从 {src_height}x{src_width} resize 到 "
                f"{dst_height}x{dst_width}，建议 engine 与输入分辨率一致"
            )
            left_img = cv2.resize(left_img, (dst_width, dst_height))
            right_img = cv2.resize(right_img, (dst_width, dst_height))
            return left_img, right_img, (dst_width / src_width, dst_height / src_height)
        return left_img, right_img, None

    def infer(self, left_img, right_img):
        height, width = left_img.shape[:2]
        img0 = torch.as_tensor(left_img, device="cuda").float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_img, device="cuda").float()[None].permute(0, 3, 1, 2)
        disp = self.model.forward(img0, img1)
        return disp.data.cpu().numpy().reshape(height, width).clip(0, None)


class StereoDepthNode:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()

        self._left_msg = None
        self._right_msg = None
        self._buf_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._last_processed_stamp = None

        self.K = None
        self.baseline = None
        # 深度配准：left_ir → color
        self.K_color = None
        self.T_ir2color = None  # 4x4 变换矩阵

        self._frame_count = 0
        self._total_infer_time = 0.0
        self._last_log_time = 0.0

        self.runner = self._build_runner()
        self._setup_ros()

        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()
        rospy.loginfo("StereoDepthNode 就绪")

    def _build_runner(self):
        if self.args.backend == "pytorch":
            return PytorchBackendRunner(self.args)
        if self.args.backend == "tensorrt":
            return TensorRTBackendRunner(self.args)
        raise ValueError(f"不支持的 backend: {self.args.backend}")

    def _setup_ros(self):
        self.depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=1)
        self.depth_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        self.disp_pub = rospy.Publisher("/camera/depth/disparity", Image, queue_size=1)
        self.pc_pub = rospy.Publisher("/camera/depth/points", PointCloud2, queue_size=1)

        self._get_camera_info()
        rospy.Subscriber(self.args.left_topic, Image, self._left_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.args.right_topic, Image, self._right_cb, queue_size=1, buff_size=2**24)

    def _get_camera_info(self):
        rospy.loginfo("等待 camera_info...")
        left_info = rospy.wait_for_message(self.args.left_info_topic, CameraInfo, timeout=15.0)
        right_info = rospy.wait_for_message(self.args.right_info_topic, CameraInfo, timeout=15.0)

        self.K = np.array(left_info.K, dtype=np.float32).reshape(3, 3)
        fx = self.K[0, 0]

        p_right = np.array(right_info.P, dtype=np.float64).reshape(3, 4)
        tx = p_right[0, 3]
        self.baseline = abs(tx) / fx

        if self.baseline < 1e-6:
            rospy.logwarn(f"基线异常: {self.baseline:.6f}m, P_right={right_info.P}")

        rospy.loginfo(
            f"Left IR 内参 fx={fx:.2f} fy={self.K[1, 1]:.2f} cx={self.K[0, 2]:.2f} "
            f"cy={self.K[1, 2]:.2f} | 基线={self.baseline:.4f}m"
        )

        # 深度配准：获取 RGB 内参和 left_ir → color 变换
        if self.args.get("depth_registration", 0):
            color_info = rospy.wait_for_message(self.args.color_info_topic, CameraInfo, timeout=15.0)
            self.K_color = np.array(color_info.K, dtype=np.float32).reshape(3, 3)

            # 从 TF 获取 left_ir_optical → color_optical 的变换
            tf_buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(tf_buffer)
            rospy.sleep(1.0)  # 等待 TF 缓冲
            try:
                trans = tf_buffer.lookup_transform(
                    "camera_color_optical_frame",
                    "camera_left_ir_optical_frame",
                    rospy.Time(0),
                    rospy.Duration(5.0),
                )
                t = trans.transform.translation
                q = trans.transform.rotation
                self.T_ir2color = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
                self.T_ir2color[:3, 3] = [t.x, t.y, t.z]
            except Exception as e:
                rospy.logerr(f"获取 TF 变换失败: {e}，禁用深度配准")
                self.T_ir2color = None

            if self.T_ir2color is not None:
                # 预计算配准常量（GPU tensor），避免每帧重复计算和 CPU-GPU 传输
                H, W = left_info.height, left_info.width
                self._reg_R = torch.from_numpy(self.T_ir2color[:3, :3].astype(np.float32)).cuda()
                self._reg_t = torch.from_numpy(self.T_ir2color[:3, 3].astype(np.float32)).cuda()
                self._reg_u_norm = torch.from_numpy(
                    ((np.arange(W, dtype=np.float32) - self.K[0, 2]) / self.K[0, 0])
                ).cuda()
                self._reg_v_norm = torch.from_numpy(
                    ((np.arange(H, dtype=np.float32) - self.K[1, 2]) / self.K[1, 1])
                ).cuda()
                self._reg_fx_c = float(self.K_color[0, 0])
                self._reg_fy_c = float(self.K_color[1, 1])
                self._reg_cx_c = float(self.K_color[0, 2])
                self._reg_cy_c = float(self.K_color[1, 2])
                self._reg_H = H
                self._reg_W = W

                rospy.loginfo(
                    f"RGB 内参 fx={self.K_color[0,0]:.2f} fy={self.K_color[1,1]:.2f} "
                    f"cx={self.K_color[0,2]:.2f} cy={self.K_color[1,2]:.2f} | "
                    f"IR→Color 平移=[{self.T_ir2color[0,3]*1000:.1f}, "
                    f"{self.T_ir2color[1,3]*1000:.1f}, {self.T_ir2color[2,3]*1000:.1f}]mm"
                )

        # 构建 depth camera_info 模板（配准后用 RGB 内参，否则用 IR 内参）
        K_out = self.K_color if self.T_ir2color is not None else self.K
        frame_out = "camera_color_optical_frame" if self.T_ir2color is not None else self.args.frame_id
        self._depth_camera_info = CameraInfo()
        self._depth_camera_info.header.frame_id = frame_out
        self._depth_camera_info.height = left_info.height
        self._depth_camera_info.width = left_info.width
        self._depth_camera_info.distortion_model = "plumb_bob"
        self._depth_camera_info.D = [0.0] * 5
        self._depth_camera_info.K = K_out.flatten().tolist()
        self._depth_camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self._depth_camera_info.P = [
            K_out[0, 0], 0, K_out[0, 2], 0,
            0, K_out[1, 1], K_out[1, 2], 0,
            0, 0, 1, 0,
        ]

    def _left_cb(self, msg):
        with self._buf_lock:
            self._left_msg = msg
        self._new_frame_event.set()

    def _right_cb(self, msg):
        with self._buf_lock:
            self._right_msg = msg
        self._new_frame_event.set()

    def _grab_latest_pair(self):
        """取最新的已配对帧，返回 (left_msg, right_msg) 或 (None, None)。"""
        with self._buf_lock:
            left_msg = self._left_msg
            right_msg = self._right_msg
        if left_msg is None or right_msg is None:
            return None, None
        if abs((left_msg.header.stamp - right_msg.header.stamp).to_sec()) > 0.001:
            return None, None
        if left_msg.header.stamp == self._last_processed_stamp:
            return None, None
        return left_msg, right_msg

    def _process_loop(self):
        while not rospy.is_shutdown():
            self._new_frame_event.wait(timeout=1.0)
            self._new_frame_event.clear()

            # 处理完后再取一次最新帧，跳过处理期间堆积的旧帧
            left_msg, right_msg = self._grab_latest_pair()
            if left_msg is None:
                continue

            self._last_processed_stamp = left_msg.header.stamp

            has_depth = self.depth_pub.get_num_connections() > 0
            has_disp = self.disp_pub.get_num_connections() > 0
            has_pc = self.pc_pub.get_num_connections() > 0
            if not (has_depth or has_disp or has_pc):
                continue

            self._infer_and_publish(left_msg, right_msg, has_depth, has_disp, has_pc)

    def _denoise_depth(self, depth, zmin):
        """
        深度图去噪：
        1. 双边滤波保边平滑，移除深度跳变噪声
        2. 孤立点移除：如果某像素周围有效深度点太稀疏，置零
        """
        valid = depth > zmin
        if not valid.any():
            return depth

        # 双边滤波：保留边缘的同时平滑深度噪声
        # d=5 邻域大小, sigmaColor 控制深度差异容忍度, sigmaSpace 控制空间范围
        depth_filtered = cv2.bilateralFilter(depth, d=5, sigmaColor=0.05, sigmaSpace=5)
        # 只对有效区域应用滤波结果
        depth_out = np.where(valid, depth_filtered, 0.0).astype(np.float32)

        # 孤立点移除：周围有效深度比例太低的点视为噪声
        kernel_size = int(self.args.get("denoise_kernel", 15))
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        valid_ratio = cv2.blur(
            (depth_out > zmin).astype(np.float32),
            (kernel_size, kernel_size),
        )
        noise_mask = (depth_out > zmin) & (valid_ratio < 0.3)
        depth_out[noise_mask] = 0.0

        return depth_out

    def _filter_edge_flying_pixels(self, depth, zmin):
        """
        移除深度不连续边缘的飞点。
        在物体边缘，立体匹配会产生前景/背景之间的错误中间值，
        投影到 3D 后形成沿射线方向的杂点。
        检测深度梯度相对于深度值的比率，超过阈值的像素置零。
        """
        threshold = float(self.args.get("edge_threshold", 0.05))
        valid = depth > zmin
        if not valid.any():
            return depth

        # 计算 x/y 方向的深度梯度
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 梯度与深度的比值，归一化消除远近物体的差异
        ratio = np.zeros_like(depth)
        ratio[valid] = grad_mag[valid] / depth[valid]

        # 比值超过阈值 → 边缘飞点，膨胀 1 像素确保边缘完全覆盖
        edge_mask = ratio > threshold
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

        depth_out = depth.copy()
        depth_out[edge_mask] = 0.0
        return depth_out

    def _register_depth_to_color(self, depth_ir):
        """
        GPU 深度配准：left IR → RGB 坐标系。
        全程 GPU 计算，scatter_reduce_ 做并行 z-buffer。
        """
        H, W = self._reg_H, self._reg_W
        if not isinstance(depth_ir, np.ndarray):
            depth_ir = np.asarray(depth_ir)
        depth_t = torch.as_tensor(depth_ir, device="cuda", dtype=torch.float32)
        valid = depth_t > 0
        if not valid.any():
            return np.zeros((H, W), dtype=np.float32)

        v_idx, u_idx = torch.where(valid)
        z = depth_t[v_idx, u_idx]

        # IR 像素 → 3D（预计算归一化坐标）→ color 坐标系
        x_ir = self._reg_u_norm[u_idx] * z
        y_ir = self._reg_v_norm[v_idx] * z
        R, t = self._reg_R, self._reg_t
        x_c = R[0, 0] * x_ir + R[0, 1] * y_ir + R[0, 2] * z + t[0]
        y_c = R[1, 0] * x_ir + R[1, 1] * y_ir + R[1, 2] * z + t[1]
        z_c = R[2, 0] * x_ir + R[2, 1] * y_ir + R[2, 2] * z + t[2]

        # 投影到 RGB 像素并过滤越界
        inv_z = 1.0 / z_c
        u_c = torch.round(self._reg_fx_c * x_c * inv_z + self._reg_cx_c).long()
        v_c = torch.round(self._reg_fy_c * y_c * inv_z + self._reg_cy_c).long()
        in_bounds = (z_c > 0) & (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
        linear_idx = v_c[in_bounds] * W + u_c[in_bounds]
        z_valid = z_c[in_bounds]

        # GPU 并行 z-buffer：scatter_reduce 取最小深度
        depth_flat = torch.full((H * W,), float("inf"), device="cuda", dtype=torch.float32)
        depth_flat.scatter_reduce_(0, linear_idx, z_valid.float(), reduce="amin", include_self=True)
        depth_flat[torch.isinf(depth_flat)] = 0.0

        return depth_flat.reshape(H, W).cpu().numpy()

    def _infer_and_publish(self, left_msg, right_msg, pub_depth, pub_disp, pub_pc):
        t_start = time.perf_counter()
        # 取帧延迟：从相机采集到开始处理的时间
        grab_lag = (rospy.Time.now() - left_msg.header.stamp).to_sec()

        # --- 预处理 ---
        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="passthrough")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="passthrough")
        left_img = to_rgb_image(left_img)
        right_img = to_rgb_image(right_img)
        left_prepared, right_prepared, scale = self.runner.prepare_images(left_img, right_img)
        left_color = left_prepared
        height, width = left_prepared.shape[:2]
        t_pre = time.perf_counter()

        # --- 模型推理 ---
        disp_np = self.runner.infer(left_prepared, right_prepared)
        torch.cuda.synchronize()
        t_infer = time.perf_counter()

        # --- 后处理 ---
        if scale is not None:
            K = self.K.copy()
            if isinstance(scale, tuple):
                K[0, :] *= scale[0]
                K[1, :] *= scale[1]
            else:
                K[:2] *= scale
        else:
            K = self.K

        # 移除右视图不可见区域
        if self.args.get("remove_invisible", 0):
            H_d, W_d = disp_np.shape
            xx = np.arange(W_d)[None, :].repeat(H_d, axis=0)
            invalid = (xx - disp_np) < 0
            disp_np[invalid] = np.inf
        t_post_invis = time.perf_counter()

        fx = K[0, 0]
        zfar = float(self.args.zfar)
        zmin = float(self.args.get("zmin", 0.1))
        safe_disp = np.where(disp_np > 0.5, disp_np, np.inf)
        depth = (fx * self.baseline / safe_disp).astype(np.float32)
        depth[depth > zfar] = 0.0
        depth[depth < zmin] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        t_post_depth = time.perf_counter()

        # 深度去噪
        if self.args.get("denoise_cloud", 0):
            depth = self._denoise_depth(depth, zmin)
        t_post_denoise = time.perf_counter()

        # 边缘飞点过滤
        if self.args.get("edge_filter", 0):
            depth = self._filter_edge_flying_pixels(depth, zmin)
        t_post_edge = time.perf_counter()

        # 深度配准：left IR → RGB
        if self.T_ir2color is not None:
            depth = self._register_depth_to_color(depth)
            K = self.K_color
            frame_id = "camera_color_optical_frame"
        else:
            frame_id = self.args.frame_id
        t_post = time.perf_counter()

        # --- 发布 ---
        stamp = left_msg.header.stamp

        if pub_depth:
            if self.args.get("depth_format", "32fc1_m") == "16uc1_mm":
                depth_pub_img = (depth * 1000).clip(0, 65535).astype(np.uint16)
                depth_msg = self.bridge.cv2_to_imgmsg(depth_pub_img, encoding="16UC1")
            else:
                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = frame_id
            self.depth_pub.publish(depth_msg)

            # 同步发布 camera_info
            info_msg = self._depth_camera_info
            info_msg.header.stamp = stamp
            self.depth_info_pub.publish(info_msg)

        if pub_disp:
            disp_msg = self.bridge.cv2_to_imgmsg(disp_np.astype(np.float32), encoding="32FC1")
            disp_msg.header.stamp = stamp
            disp_msg.header.frame_id = frame_id
            self.disp_pub.publish(disp_msg)

        if pub_pc:
            xyz_map = depth2xyzmap(depth, K, zmin=zmin)
            pts = xyz_map.reshape(-1, 3)
            colors = left_color.reshape(-1, 3)
            valid = (pts[:, 2] > zmin) & (pts[:, 2] <= zfar)

            header = Header(stamp=stamp, frame_id=frame_id)
            pc_msg = build_point_cloud2(
                header,
                pts[valid].astype(np.float32),
                colors[valid].astype(np.uint8),
            )
            self.pc_pub.publish(pc_msg)
        t_pub = time.perf_counter()

        # --- 统计 ---
        dt_pre = (t_pre - t_start) * 1000
        dt_infer = (t_infer - t_pre) * 1000
        dt_invis = (t_post_invis - t_infer) * 1000
        dt_depth = (t_post_depth - t_post_invis) * 1000
        dt_denoise = (t_post_denoise - t_post_depth) * 1000
        dt_edge = (t_post_edge - t_post_denoise) * 1000
        dt_reg = (t_post - t_post_edge) * 1000
        dt_pub = (t_pub - t_post) * 1000
        dt_total = (t_pub - t_start) * 1000

        self._frame_count += 1
        self._total_infer_time += dt_total / 1000
        now = time.perf_counter()
        if now - self._last_log_time >= 5.0:
            avg = self._total_infer_time / self._frame_count * 1000
            end_lag = (rospy.Time.now() - stamp).to_sec()
            rospy.loginfo(
                f"[统计] 帧#{self._frame_count} | "
                f"预处理 {dt_pre:.1f} | 推理 {dt_infer:.1f} | "
                f"去遮挡 {dt_invis:.1f} | 转深度 {dt_depth:.1f} | 去噪 {dt_denoise:.1f} | "
                f"去飞点 {dt_edge:.1f} | 配准 {dt_reg:.1f} | "
                f"发布 {dt_pub:.1f} | 总计 {dt_total:.0f}ms | "
                f"平均 {avg:.0f}ms ({1000/avg:.1f}FPS) | "
                f"取帧延迟 {grab_lag*1000:.0f}ms | 端到端 {end_lag*1000:.0f}ms"
            )
            self._last_log_time = now


def main():
    parser = build_parser()
    cli_args, _ = parser.parse_known_args()
    args = merge_config(cli_args)

    ensure_ros_imported()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    logging.info("effective args:\n%s", OmegaConf.to_yaml(args))

    rospy.init_node("foundation_stereo_depth", anonymous=True)
    node = StereoDepthNode(args)
    rospy.spin()


if __name__ == "__main__":
    main()
