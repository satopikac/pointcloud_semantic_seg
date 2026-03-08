#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 需要修改
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3, Quaternion
from std_msgs.msg import ColorRGBA, Header
import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLOE
import torch
import open3d as o3d
from scipy.optimize import linear_sum_assignment
import json
import os
import trimesh  # 用于3D IoU计算
from threading import Lock
from scipy.spatial.transform import Rotation as R_scipy
# ==================== 点云处理工具 ====================
class PointCloudUtils:
    @staticmethod
    def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
        """统计滤波去除离群点"""
        if len(pcd.points) < nb_neighbors:
            return pcd
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        return pcd.select_by_index(ind)

    @staticmethod
    def voxel_downsample(pcd, voxel_size):
        """体素降采样"""
        if voxel_size <= 0:
            return pcd
        return pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def compute_obb(pcd):
        """计算定向包围盒"""
        if len(pcd.points) < 4:
            return None
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        # 旋转矩阵 -> 四元数
        R = obb.R  # 从 Open3D 获取的矩阵可能是只读的
        from scipy.spatial.transform import Rotation as R_scipy
        # 复制矩阵以确保可写
        r = R_scipy.from_matrix(np.array(R, copy=True))
        quat = r.as_quat()  # [x, y, z, w]
        extent = obb.extent
        return center, extent, quat

    @staticmethod
    def compute_iou_3d(obb1_center, obb1_extent, obb1_quat,
                       obb2_center, obb2_extent, obb2_quat):
        """计算两个OBB的3D IoU（使用trimesh）"""
        # 构建第一个长方体的变换矩阵
        T1 = np.eye(4)
        T1[:3, 3] = obb1_center
        R1 = R_scipy.from_quat(obb1_quat).as_matrix()
        T1[:3, :3] = R1
        mesh1 = trimesh.primitives.Box(extents=obb1_extent, transform=T1)

        T2 = np.eye(4)
        T2[:3, 3] = obb2_center
        R2 = R_scipy.from_quat(obb2_quat).as_matrix()
        T2[:3, :3] = R2
        mesh2 = trimesh.primitives.Box(extents=obb2_extent, transform=T2)

        try:
            intersection = mesh1.intersection(mesh2)
            vol_intersection = intersection.volume
        except:
            vol_intersection = 0.0
        vol1 = mesh1.volume
        vol2 = mesh2.volume
        iou = vol_intersection / (vol1 + vol2 - vol_intersection + 1e-10)
        return iou


# ==================== 物体模型 ====================
class Object3D:
    def __init__(self, obj_id, label, pointcloud, color=None):
        self.id = obj_id
        self.label = label
        self.pointcloud = pointcloud  # open3d点云
        self.color = color if color is not None else self._id_to_color(obj_id)
        self.last_seen = rospy.Time.now()
        self.completed = False
        self.center = None
        self.extent = None
        self.quat = None  # 四元数 (x,y,z,w)
        self.update_obb()

    def _id_to_color(self, obj_id):
        """根据ID生成颜色"""
        np.random.seed(obj_id)
        return np.random.rand(3).tolist()

    def update_obb(self):
        if len(self.pointcloud.points) >= 4:
            self.center, self.extent, self.quat = PointCloudUtils.compute_obb(self.pointcloud)
        else:
            self.center = self.extent = self.quat = None

    def to_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'center': self.center.tolist() if self.center is not None else None,
            'extent': self.extent.tolist() if self.extent is not None else None,
            'quat': self.quat.tolist() if self.quat is not None else None,
            'last_seen': self.last_seen.to_sec()
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(data['id'], data['label'], o3d.geometry.PointCloud())
        obj.completed = True
        obj.center = np.array(data['center'])
        obj.extent = np.array(data['extent'])
        obj.quat = np.array(data['quat'])
        obj.last_seen = rospy.Time(data['last_seen'])
        return obj


# ==================== 持久化 ====================
class Persistence:
    def __init__(self, filename='objects.json'):
        self.filename = filename
        self.lock = Lock()

    def load(self):
        if not os.path.exists(self.filename):
            return []
        with self.lock:
            with open(self.filename, 'r') as f:
                data = json.load(f)
        return [Object3D.from_dict(d) for d in data]

    def save(self, objects):
        data = [obj.to_dict() for obj in objects if obj.completed]
        with self.lock:
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)


# ==================== 匈牙利匹配 ====================
class HungarianMatcher:
    def __init__(self, iou_threshold=0.1):
        self.iou_threshold = iou_threshold

    def match(self, detections, tracked_objects):
        """输入：当前检测的OBB信息列表，跟踪的物体列表；返回匹配对 (det_idx, obj_id)"""
        n_det = len(detections)
        n_trk = len(tracked_objects)
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))

        # 构建代价矩阵：负IoU（匈牙利求最小代价）
        cost_matrix = np.full((n_det, n_trk), 1e6)
        for i, det in enumerate(detections):
            for j, obj in enumerate(tracked_objects):
                if obj.completed:
                    continue  # 已完成物体不参与匹配（但可以从JSON加载的物体也可以？暂时不考虑）
                iou = PointCloudUtils.compute_iou_3d(
                    det['center'], det['extent'], det['quat'],
                    obj.center, obj.extent, obj.quat
                )
                if iou >= self.iou_threshold:
                    cost_matrix[i, j] = -iou  # 取负值使匈牙利最小化

        # 执行匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_det = list(range(n_det))
        unmatched_trk = list(range(n_trk))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:  # 有效匹配
                matches.append((r, c))
                if r in unmatched_det:
                    unmatched_det.remove(r)
                if c in unmatched_trk:
                    unmatched_trk.remove(c)
        return matches, unmatched_det, unmatched_trk


# ==================== YOLOE检测器 ====================
class YOLOEDetector:
    def __init__(self, model_path, conf_threshold, device):
        self.model = YOLOE(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_names = self.model.names  # 所有类别名称

    def detect(self, image):
        """返回每个实例的掩码（bool）、类别ID、置信度"""
        results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)
        if not results or len(results) == 0:
            return []
        result = results[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy()  # (N, H, W) float
        masks = masks > 0.5
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        detections = []
        for i in range(len(masks)):
            detections.append({
                'mask': masks[i],
                'class_id': classes[i],
                'confidence': confs[i],
                'label': self.class_names[classes[i]]
            })
        return detections


# ==================== 传感器数据处理 ====================
class SensorProcessor:
    def __init__(self, target_frame, max_depth):
        self.target_frame = target_frame
        self.max_depth = max_depth
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.camera_info = None
        self.K = None

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.K = np.array(msg.K).reshape(3, 3)

    def process(self, rgb_msg, depth_msg):
        # 转换图像
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        if self.K is None:
            rospy.logwarn_throttle(5, "Camera info not received yet.")
            return None, None, None

        # 获取变换
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, rgb_msg.header.frame_id, rgb_msg.header.stamp, rospy.Duration(0.1))
            T = self._transform_to_matrix(transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF error: {e}")
            return None, None, None

        return rgb, depth, T

    def _transform_to_matrix(self, transform):
        from tf.transformations import quaternion_matrix
        t = transform.transform.translation
        q = transform.transform.rotation
        mat = quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat

    def project_mask(self, mask, depth, T, rgb_image):
        """反投影掩码内的像素到3D，并获取颜色"""
        v_coords, u_coords = np.where(mask)
        if len(u_coords) == 0:
            return None, None
        depths = depth[v_coords, u_coords]
        valid = (depths > 0) & (depths < self.max_depth)
        u_valid = u_coords[valid]
        v_valid = v_coords[valid]
        d_valid = depths[valid]
        if len(u_valid) == 0:
            return None, None

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        Z_cam = d_valid
        X_cam = (u_valid - cx) * Z_cam / fx
        Y_cam = (v_valid - cy) * Z_cam / fy
        points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)

        # 转换到目标坐标系
        ones = np.ones((points_cam.shape[0], 1))
        points_homo = np.hstack([points_cam, ones])
        points_global = (T @ points_homo.T).T[:, :3]

        # 获取颜色
        colors = rgb_image[v_valid, u_valid, :] / 255.0  # 归一化到[0,1]

        return points_global, colors


# ==================== 可视化 ====================
class Visualizer:
    def __init__(self, target_frame):
        self.target_frame = target_frame
        self.marker_pub = rospy.Publisher('/detected_objects_markers', MarkerArray, queue_size=10)
        self.pc_pub = rospy.Publisher('/object_pointclouds', PointCloud2, queue_size=10)

    def publish(self, objects):
        now = rospy.Time.now()
        marker_array = MarkerArray()
        # 合并所有物体的点云（彩色）
        all_points = []
        all_colors = []
        for obj in objects:
            if obj.completed:
                color = [0.5, 0.5, 0.5]  # 灰色
            else:
                color = obj.color
            # 如果有点云数据
            if len(obj.pointcloud.points) > 0:
                pts = np.asarray(obj.pointcloud.points)
                all_points.append(pts)
                # 为每个点分配颜色
                colors = np.tile(color, (pts.shape[0], 1))
                all_colors.append(colors)

            # 发布立方体标记
            if obj.center is not None:
                marker = Marker()
                marker.header.frame_id = self.target_frame
                marker.header.stamp = now
                marker.ns = "cubes"
                marker.id = obj.id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = obj.center[0]
                marker.pose.position.y = obj.center[1]
                marker.pose.position.z = obj.center[2]
                marker.pose.orientation.x = obj.quat[0]
                marker.pose.orientation.y = obj.quat[1]
                marker.pose.orientation.z = obj.quat[2]
                marker.pose.orientation.w = obj.quat[3]
                marker.scale.x = obj.extent[0]
                marker.scale.y = obj.extent[1]
                marker.scale.z = obj.extent[2]
                if obj.completed:
                    marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.5)  # 灰色半透明
                else:
                    marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)  # 绿色半透明
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)

                # 标签
                text_marker = Marker()
                text_marker.header.frame_id = self.target_frame
                text_marker.header.stamp = now
                text_marker.ns = "labels"
                text_marker.id = obj.id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = obj.center[0]
                text_marker.pose.position.y = obj.center[1]
                text_marker.pose.position.z = obj.center[2] + obj.extent[2]/2 + 0.3
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.2
                text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
                text_marker.text = f"{obj.label}_{obj.id}"
                text_marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

        # 发布合并点云
        if all_points:
            points = np.vstack(all_points)
            colors = np.vstack(all_colors)
            self._publish_pointcloud(points, colors, now)

    def _publish_pointcloud(self, points, colors, stamp):
        """发布彩色点云"""
        from sensor_msgs.msg import PointCloud2, PointField
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        # 构造数据
        data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('r', np.float32), ('g', np.float32), ('b', np.float32)
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['r'] = colors[:, 0]
        data['g'] = colors[:, 1]
        data['b'] = colors[:, 2]

        pc_msg = PointCloud2()
        pc_msg.header.stamp = stamp
        pc_msg.header.frame_id = self.target_frame
        pc_msg.height = 1
        pc_msg.width = points.shape[0]
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 24  # 4 bytes per field * 6 fields
        pc_msg.row_step = pc_msg.point_step * points.shape[0]
        pc_msg.is_dense = True
        pc_msg.data = data.tobytes()
        self.pc_pub.publish(pc_msg)


# ==================== 主节点 ====================
class ObjectMapperNode:
    def __init__(self):
        rospy.init_node('object_mapper', anonymous=False)

        # 参数
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.target_classes = rospy.get_param('~target_classes', None)  # None表示使用模型默认所有类别
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.max_depth = rospy.get_param('~max_depth', 2)
        self.voxel_size = rospy.get_param('~voxel_size', 0.05)
        self.outlier_nb_neighbors = rospy.get_param('~outlier_nb_neighbors', 20)
        self.outlier_std_ratio = rospy.get_param('~outlier_std_ratio', 2.0)
        self.association_threshold = rospy.get_param('~association_threshold', 1.0)
        self.iou_threshold = rospy.get_param('~iou_threshold', 0.1)
        self.timeout_sec = rospy.get_param('~timeout_sec', 30.0)
        self.model_path = rospy.get_param('~model_path', 'yoloe-26l-seg-pf.pt')
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.json_file = rospy.get_param('~json_file', 'objects.json')

        # 初始化模块
        self.sensor = SensorProcessor(self.target_frame, self.max_depth)
        self.detector = YOLOEDetector(self.model_path, self.conf_threshold, self.device)
        self.matcher = HungarianMatcher(self.iou_threshold)
        self.persistence = Persistence(self.json_file)
        self.visualizer = Visualizer(self.target_frame)

        # 加载已保存的物体
        self.objects = {}  # id -> Object3D
        for obj in self.persistence.load():
            self.objects[obj.id] = obj
            self.visualizer.publish(list(self.objects.values()))  # 立即显示

        # 数据同步
        self.sub_rgb = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.sub_depth = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.sub_info = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # 定时器：更新状态和发布
        self.timer = rospy.Timer(rospy.Duration(0.2), self.update_and_publish)

        rospy.loginfo("Object mapper started.")

    def callback(self, rgb_msg, depth_msg, info_msg):
        """处理每一帧图像和深度数据"""
        try:
            # ========== 1. 传感器数据处理 ==========
            self.sensor.camera_info_callback(info_msg)
            rgb, depth, T = self.sensor.process(rgb_msg, depth_msg)
            if rgb is None:
                return

            # ========== 2. YOLOE 检测 ==========
            detections = self.detector.detect(rgb)
            if not detections:
                return

            now = rospy.Time.now()
            temp_objects = []  # 存储当前帧的检测结果（点云+OBB）

            # ========== 3. 对每个检测实例生成点云 ==========
            for det in detections:
                mask = det['mask']
                label = det['label']
                # 反投影掩码内的像素到3D并获取颜色
                points, colors = self.sensor.project_mask(mask, depth, T, rgb)
                if points is None or len(points) < 10:  # 最少点数
                    continue

                # 创建 Open3D 点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                # 统计滤波去除离群点
                pcd = PointCloudUtils.statistical_outlier_removal(
                    pcd, self.outlier_nb_neighbors, self.outlier_std_ratio)
                if len(pcd.points) < 10:
                    continue

                # 体素降采样
                pcd = PointCloudUtils.voxel_downsample(pcd, self.voxel_size)
                if len(pcd.points) < 10:
                    continue

                # 计算定向包围盒 OBB
                obb_result = PointCloudUtils.compute_obb(pcd)
                if obb_result is None:
                    continue
                center, extent, quat = obb_result

                temp_objects.append({
                    'pcd': pcd,
                    'label': label,
                    'center': center,
                    'extent': extent,
                    'quat': quat,
                    'color': None
                })

            if not temp_objects:
                return

            # ========== 类型安全检查：确保 self.objects 中全是 Object3D ==========
            for obj_id, obj in list(self.objects.items()):
                if not isinstance(obj, Object3D):
                    rospy.logwarn_throttle(10, f"Object {obj_id} has invalid type {type(obj)}, removing.")
                    del self.objects[obj_id]

            # ========== 4. 获取已完成物体 ==========
            completed_objects = [obj for obj in self.objects.values() if obj.completed]

            # ========== 5. 过滤掉与已完成物体重叠的检测 ==========
            filtered_detections = []
            for det in temp_objects:
                matched_completed = False
                for comp_obj in completed_objects:
                    iou = PointCloudUtils.compute_iou_3d(
                        det['center'], det['extent'], det['quat'],
                        comp_obj.center, comp_obj.extent, comp_obj.quat
                    )
                    if iou >= self.iou_threshold:
                        matched_completed = True
                        rospy.logdebug(f"Detection matches completed object {comp_obj.id}, ignored.")
                        break
                if not matched_completed:
                    filtered_detections.append(det)

            if not filtered_detections:
                return

            # ========== 6. 获取当前活动物体（未完成） ==========
            active_objects = [obj for obj in self.objects.values() if not obj.completed]

            # ========== 7. 匈牙利匹配（基于3D IoU） ==========
            det_obbs = [{'center': d['center'], 'extent': d['extent'], 'quat': d['quat']} 
                        for d in filtered_detections]
            tracked_obbs = [{'center': obj.center, 'extent': obj.extent, 'quat': obj.quat} 
                            for obj in active_objects]

            matches, unmatched_det_idx, unmatched_trk_idx = self.matcher.match(det_obbs, tracked_obbs)

            # ========== 8. 处理匹配成功的检测（融合点云） ==========
            for det_idx, trk_idx in matches:
                obj = active_objects[trk_idx]
                det = filtered_detections[det_idx]
                # 合并点云
                combined = obj.pointcloud + det['pcd']
                obj.pointcloud = PointCloudUtils.voxel_downsample(combined, self.voxel_size)
                obj.update_obb()
                obj.last_seen = now

            # ========== 9. 处理未匹配的检测（创建新物体） ==========
            for det_idx in unmatched_det_idx:
                det = filtered_detections[det_idx]
                obj_id = len(self.objects)
                new_obj = Object3D(obj_id, det['label'], det['pcd'])
                self.objects[obj_id] = new_obj
                rospy.loginfo(f"New object {obj_id} created.")

            # ========== 10. 处理未匹配的跟踪物体（检查超时） ==========
            for trk_idx in unmatched_trk_idx:
                obj = active_objects[trk_idx]
                if (now - obj.last_seen).to_sec() > self.timeout_sec:
                    obj.completed = True
                    rospy.loginfo(f"Object {obj.id} completed and saved.")
                    self.persistence.save(list(self.objects.values()))

        except Exception as e:
            rospy.logerr_throttle(10, f"Callback error: {e}")

    def update_and_publish(self, event):
        """定期更新超时状态并发布可视化"""
        now = rospy.Time.now()
        for obj in list(self.objects.values()):
            if not obj.completed and (now - obj.last_seen).to_sec() > self.timeout_sec:
                obj.completed = True
                rospy.loginfo(f"Object {obj.id} completed (timeout).")
                self.persistence.save(list(self.objects.values()))

        # 发布所有物体（包括已完成的）
        self.visualizer.publish(list(self.objects.values()))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ObjectMapperNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
