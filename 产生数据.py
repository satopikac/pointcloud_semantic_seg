#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import rospy
import rosbag
import cv2
import message_filters
import tf2_ros
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped

# --- [!!! 用户配置 - 发布节点 !!!] ---
BAG_FILE_PATH = "../2025-12-01-19-01-56.bag"
TARGET_FRAME_INDEX_START = 50  # 初始帧索引
FRAME_ID = "camera_color_frame"
MAP_FRAME_ID = "map"
TIME_TOLERANCE = rospy.Duration(0.01)

# 动态控制参数
FRAME_INDEX_INCREMENT = 50  # 每次跳跃的帧数
CYCLE_RATE_SEC = 10  # 每 10 秒切换一次帧
PUBLISH_RATE_HZ = 10  # 持续发布的频率
PUB_TOPIC_RGB = "/input/frame_50_rgb"
PUB_TOPIC_DEPTH = "/input/frame_50_depth"
PUB_TOPIC_INFO = "/input/frame_50_info"
PUB_TOPIC_PCD_RAW = "/input/frame_50_raw_rgb_pcd"
DEPTH_SCALE = 1000.0
# ----------------------------


class BagFramePublisher:
    def __init__(self):
        rospy.init_node("bag_frame_publisher", anonymous=True, log_level=rospy.INFO)
        self.bridge = CvBridge()

        # 帧控制变量
        self.sync_frames_list = []  # 存储所有提取的同步消息列表
        self.max_index = 0
        self.current_index = TARGET_FRAME_INDEX_START

        # 内存中存储的消息 (当前发布的帧数据)
        self.rgb_msg = None
        self.depth_msg = None
        self.info_msg = None
        self.pc_raw_msg = None

        # 传感器发布器
        self.pub_rgb = rospy.Publisher(PUB_TOPIC_RGB, Image, queue_size=1)
        self.pub_depth = rospy.Publisher(PUB_TOPIC_DEPTH, Image, queue_size=1)
        self.pub_info = rospy.Publisher(PUB_TOPIC_INFO, CameraInfo, queue_size=1)
        self.pub_pcd_raw = rospy.Publisher(PUB_TOPIC_PCD_RAW, PointCloud2, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 1. 提取所有同步数据到内存
        if not self._load_all_sync_frames():
            rospy.signal_shutdown("无法加载数据，节点退出。")
            return

        # 2. 初始化第一帧数据
        self._update_current_frame_data()

        # 3. 启动定时器，每 CYCLE_RATE_SEC 秒更新一次帧索引
        rospy.Timer(rospy.Duration(CYCLE_RATE_SEC), self._cycle_timer_callback)
        rospy.loginfo(f"数据循环定时器启动，每 {CYCLE_RATE_SEC} 秒切换下一帧。")

    def _load_all_sync_frames(self):
        """从 Bag 文件中提取所有同步消息，并存储到列表中。"""
        rospy.loginfo(f"开始离线提取所有同步帧...")
        bag_topics = [
            "/camera/color/image_raw",
            "/camera/aligned_depth_to_color/image_raw",
            "/camera/color/camera_info",
        ]
        time_buffer = {}

        try:
            bag = rosbag.Bag(BAG_FILE_PATH, "r")
            for topic, msg, t in bag.read_messages(topics=bag_topics):
                time_buffer[topic] = (msg, t)

                if all(t in time_buffer for t in bag_topics):
                    t_rgb, t_depth, t_info = (
                        time_buffer[bag_topics[0]][1],
                        time_buffer[bag_topics[1]][1],
                        time_buffer[bag_topics[2]][1],
                    )

                    if (
                        abs(t_rgb - t_depth) < TIME_TOLERANCE
                        and abs(t_rgb - t_info) < TIME_TOLERANCE
                    ):

                        # 存储 (RGB Msg, Depth Msg, Info Msg)
                        frame_data = (
                            time_buffer[bag_topics[0]][0],
                            time_buffer[bag_topics[1]][0],
                            time_buffer[bag_topics[2]][0],
                        )
                        self.sync_frames_list.append(frame_data)

                        time_buffer = {}

        except Exception as e:
            rospy.logerr(f"打开或读取 Bag 文件失败: {e}")
            return False
        finally:
            bag.close()

        self.max_index = len(self.sync_frames_list)
        if self.max_index == 0:
            rospy.logwarn("未找到任何同步帧。")
            return False

        rospy.loginfo(f"成功加载 {self.max_index} 帧同步数据。")
        # 确保起始索引在有效范围内
        self.current_index = self.current_index % self.max_index
        return True

    def _encode_rgb_to_uint32(self, r: int, g: int, b: int) -> int:
        """Encodes R, G, B components into a UINT32 integer (BGR order for ROS/PCL)."""
        return (int(b) << 16) | (int(g) << 8) | int(r)

    def _generate_raw_point_cloud(self, rgb_msg, depth_msg, info_msg):
        """Converts RGB, Depth, and Info messages into a PointCloud2 message."""
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            K_matrix = info_msg.K
            fx, fy, cx, cy = K_matrix[0], K_matrix[4], K_matrix[2], K_matrix[5]
            H, W = cv_depth.shape

            points = []

            for v in range(H):
                for u in range(W):
                    depth_mm = cv_depth[v, u]
                    if depth_mm == 0:
                        continue

                    Z = float(depth_mm) / DEPTH_SCALE
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy

                    b_orig, g_orig, r_orig = cv_rgb[v, u, :].astype(int)
                    rgb_int = self._encode_rgb_to_uint32(r_orig, g_orig, b_orig)

                    points.append((X, Y, Z, rgb_int))

            pcd_fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.UINT32, 1),
            ]

            header = rospy.Header()
            header.frame_id = FRAME_ID
            return pc2.create_cloud(header, pcd_fields, points)

        except Exception as e:
            rospy.logerr(f"生成点云失败: {e}")
            return None

    def _update_current_frame_data(self):
        """根据 self.current_index 更新正在发布的消息和点云。"""
        if self.max_index == 0:
            return

        # 确保索引在有效范围内
        index_to_use = self.current_index % self.max_index

        # 从列表中取出存储的消息
        rgb_msg, depth_msg, info_msg = self.sync_frames_list[index_to_use]

        # 1. 更新存储的 ROS 消息
        self.rgb_msg = rgb_msg
        self.depth_msg = depth_msg
        self.info_msg = info_msg

        # 2. 生成点云
        self.pc_raw_msg = self._generate_raw_point_cloud(rgb_msg, depth_msg, info_msg)

        rospy.loginfo(f"--- 帧数据已更新为索引 {index_to_use} ---")

    def _cycle_timer_callback(self, event):
        """每 10 秒触发一次，更新帧索引。"""
        self.current_index += FRAME_INDEX_INCREMENT
        self._update_current_frame_data()

    def _publish_tf(self, current_time):
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = MAP_FRAME_ID
        t.child_frame_id = FRAME_ID
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def run_publish_loop(self):
        """高频发布循环，发布当前帧数据。"""
        if self.rgb_msg is None or self.pc_raw_msg is None:
            rospy.logerr("缺少数据，无法启动发布循环。")
            return

        rate = rospy.Rate(PUBLISH_RATE_HZ)
        rospy.loginfo(f"开始以 {PUBLISH_RATE_HZ} Hz 持续发布当前帧数据。")

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # 1. 更新所有 Header 的时间戳
            self.rgb_msg.header.stamp = current_time
            self.depth_msg.header.stamp = current_time
            self.info_msg.header.stamp = current_time
            self.pc_raw_msg.header.stamp = current_time

            # 2. 发布 TF 变换
            self._publish_tf(current_time)

            # 3. 发布传感器图像和内参
            self.pub_rgb.publish(self.rgb_msg)
            self.pub_depth.publish(self.depth_msg)
            self.pub_info.publish(self.info_msg)

            # 4. 发布原始点云
            self.pub_pcd_raw.publish(self.pc_raw_msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        publisher = BagFramePublisher()
        publisher.run_publish_loop()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 中断，Bag 发布节点退出。")
