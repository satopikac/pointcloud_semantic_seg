import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import message_filters

# --- [配置] ---
# 订阅话题 (程序 1 发布的话题)
SUB_TOPIC_RGB = "/input/frame_50_rgb"
SUB_TOPIC_DEPTH = "/input/frame_50_depth"
SUB_TOPIC_INFO = "/input/frame_50_info"

# 临时数据存储目录
TEMP_DATA_DIR = "/tmp/ros_mmseg_temp"
RGB_FILE = os.path.join(TEMP_DATA_DIR, "latest_rgb.png")
DEPTH_FILE = os.path.join(TEMP_DATA_DIR, "latest_depth.npy")
INFO_FILE = os.path.join(TEMP_DATA_DIR, "latest_info.json")

TIME_TOLERANCE = 0.1  # 消息同步容忍度
# -----------------


class ImageToNumpyNode:
    """订阅 ROS Image，使用 ROS 环境的 cv_bridge 转换为文件。"""

    def __init__(self):
        rospy.init_node(
            "image_to_numpy_converter", anonymous=True, log_level=rospy.INFO
        )
        self.bridge = CvBridge()
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)

        # 订阅器
        sub_rgb = message_filters.Subscriber(SUB_TOPIC_RGB, Image)
        sub_depth = message_filters.Subscriber(SUB_TOPIC_DEPTH, Image)
        sub_info = message_filters.Subscriber(SUB_TOPIC_INFO, CameraInfo)

        ats = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=5, slop=TIME_TOLERANCE
        )
        ats.registerCallback(self.image_callback)

        rospy.loginfo(f"图像转 NumPy 节点启动。数据将写入: {TEMP_DATA_DIR}")

    def image_callback(self, rgb_msg, depth_msg, info_msg):
        """同步回调，将最新数据写入临时文件。"""
        try:
            # 1. 使用 ROS 系统的 cv_bridge 进行转换 (此步骤在 Conda 中失败)
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 2. 保存 RGB (PNG) 和 深度 (NPY)
            cv2.imwrite(RGB_FILE, cv_rgb)
            np.save(DEPTH_FILE, cv_depth)

            # 3. 保存相机内参 (JSON)
            import json

            info_data = {
                "K": list(info_msg.K),
                "header": {
                    "stamp": info_msg.header.stamp.to_sec(),
                    "frame_id": info_msg.header.frame_id,
                },
            }
            with open(INFO_FILE, "w") as f:
                json.dump(info_data, f)

            rospy.logdebug(
                f"最新帧数据已更新至文件系统，时间: {rgb_msg.header.stamp.to_sec():.2f}"
            )

        except Exception as e:
            rospy.logerr(f"在 ROS 环境中转换或保存数据失败: {e}")


if __name__ == "__main__":
    try:
        node = ImageToNumpyNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点退出。")
