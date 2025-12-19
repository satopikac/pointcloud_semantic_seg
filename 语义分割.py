# 在点云新增了class_id 字段
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import rospy
import torch
import cv2
import json

# MMSegmentation 库
from mmseg.apis import init_model, inference_model 
from mmengine.config import Config
from mmseg.registry import DATASETS 
from prettytable import PrettyTable 
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
import time

# --- [配置] ---
TEMP_DATA_DIR = "/tmp/ros_mmseg_temp"
RGB_FILE = os.path.join(TEMP_DATA_DIR, "latest_rgb.png")
DEPTH_FILE = os.path.join(TEMP_DATA_DIR, "latest_depth.npy")
INFO_FILE = os.path.join(TEMP_DATA_DIR, "latest_info.json")

FRAME_ID = 'camera_color_frame' 

CONFIG_FILE = '/home/mei123/workspace/cv/mmsegmentation/config/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py'
CHECKPOINT_FILE = '/home/mei123/workspace/cv/mmsegmentation/config/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEPTH_SCALE = 1000.0 

PROCESS_RATE_SEC = 10
PUBLISH_RATE_HZ = 5
PUB_TOPIC_PCD_SEMANTIC = '/processed/semantic_point_cloud'
PUB_TOPIC_MARKER_ARRAY = '/processed/semantic_labels'
MARKER_LIFETIME = rospy.Duration(1.0 / PUBLISH_RATE_HZ * 1.5) 
MIN_POINTS_FOR_MARKER = 500 
MARKER_SCALE_Z = 0.1

VECTORIZED_BATCH_SIZE = 100000
# ----------------------------

class SemanticProcessorNode:
    def __init__(self):
        rospy.init_node('semantic_processor_node', anonymous=True, log_level=rospy.INFO)
        
        self.pc_sem_msg = None
        self.marker_array_msg = None
        self.data_ready = False 
        
        # 性能统计
        self.processing_times = []
        self.max_processing_time = 0
        self.min_processing_time = float('inf')
        
        # 依赖初始化
        self.color_cache = {} 
        self.id_to_name = self._load_model_classes()
        self.model = self._init_mmseg_model()
        
        if self.id_to_name is None or self.model is None: 
            rospy.logerr("核心依赖初始化失败，节点停止。")
            rospy.signal_shutdown("Core dependencies failed to load.")
            return

        # 发布器
        self.pub_pcd_sem = rospy.Publisher(PUB_TOPIC_PCD_SEMANTIC, PointCloud2, queue_size=1)
        self.pub_markers = rospy.Publisher(PUB_TOPIC_MARKER_ARRAY, MarkerArray, queue_size=1)

        rospy.Timer(rospy.Duration(PROCESS_RATE_SEC), self.processing_timer_callback)
        rospy.Timer(rospy.Duration(1.0 / PUBLISH_RATE_HZ), self.publish_timer_callback)
        
        rospy.loginfo(f"处理定时器 ({PROCESS_RATE_SEC}s) 启动。")
        rospy.loginfo(f"发布定时器 ({PUBLISH_RATE_HZ} Hz) 启动。")

        self._print_color_map() 
        rospy.loginfo("使用向量化点云生成优化，点云包含XYZ、RGB和Class ID五个字段")

    def _load_model_classes(self):
        try:
            cfg = Config.fromfile(CONFIG_FILE)
            from mmseg.registry import DATASETS
            DatasetClass = DATASETS.get('ADE20KDataset') 
            class_names = DatasetClass.METAINFO['classes']
            return {i: name for i, name in enumerate(class_names)}
        except Exception as e:
            rospy.logerr(f"加载类别名称失败: {e}")
            return None
            
    def _init_mmseg_model(self):
        try:
            model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
            model.eval()
            rospy.loginfo(f"MMSegmentation 模型加载成功，运行在: {DEVICE}")
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败: {e}")
            return None

    def _generate_unique_color(self, class_id: int) -> tuple:
        if class_id == 0: return (0, 0, 0)
        if class_id in self.color_cache: return self.color_cache[class_id]
        r = (class_id * 37 + 50) % 256
        g = (class_id * 101 + 100) % 256
        b = (class_id * 179 + 150) % 256
        color = (r, g, b)
        self.color_cache[class_id] = color
        return color

    def _generate_colors_batch(self, class_ids: np.ndarray) -> np.ndarray:
        """批量生成颜色"""
        unique_ids = np.unique(class_ids)
        color_map = {}
        
        for class_id in unique_ids:
            if class_id == 0:
                color_map[class_id] = (0, 0, 0)
            elif class_id in self.color_cache:
                color_map[class_id] = self.color_cache[class_id]
            else:
                r = (class_id * 37 + 50) % 256
                g = (class_id * 101 + 100) % 256
                b = (class_id * 179 + 150) % 256
                color_map[class_id] = (r, g, b)
                self.color_cache[class_id] = (r, g, b)
        
        colors = np.zeros((len(class_ids), 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            mask = class_ids == class_id
            colors[mask] = color
        
        return colors

    def _print_color_map(self):
        table = PrettyTable()
        table.field_names = ["ID", "Semantic Name (语义标签)", "RGB Color (点云颜色)"]
        for class_id in range(min(20, len(self.id_to_name))):
            r, g, b = self._generate_unique_color(class_id)
            class_name = self.id_to_name.get(class_id, "UNKNOWN")
            table.add_row([class_id, class_name, f"({r}, {g}, {b})"])
        if len(self.id_to_name) > 20:
            table.add_row(["...", "...", "..."])
            for class_id in range(len(self.id_to_name)-5, len(self.id_to_name)):
                r, g, b = self._generate_unique_color(class_id)
                class_name = self.id_to_name.get(class_id, "UNKNOWN")
                table.add_row([class_id, class_name, f"({r}, {g}, {b})"])
        rospy.loginfo("\n--- 预生成的语义点云颜色映射表 (用于 RViz 参考) ---")
        print(table)

    def _get_semantic_mask(self, cv_image: np.ndarray) -> np.ndarray:
        result = inference_model(self.model, cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        return result.pred_sem_seg.data.cpu().numpy().squeeze()

    def _encode_rgb_to_uint32(self, colors: np.ndarray) -> np.ndarray:
        """批量编码RGB到uint32"""
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        return (b << 16) | (g << 8) | r

    def _create_point_cloud_with_class_id(self, header, data_array: np.ndarray, fields):
        """创建点云消息 - 包含class_id"""
        # data_array shape: (N, 5) dtype: [x, y, z, rgb, class_id]
        data_list = []
        for i in range(len(data_array)):
            x = float(data_array[i, 0])
            y = float(data_array[i, 1])
            z = float(data_array[i, 2])
            rgb = int(data_array[i, 3])
            class_id = int(data_array[i, 4])
            data_list.append((x, y, z, rgb, class_id))
        
        return pc2.create_cloud(header, fields, data_list)

    def _create_markers(self, stamp, class_centers: dict):
        marker_array = MarkerArray()
        marker_id_counter = 0

        for class_id, data in class_centers.items():
            if data['count'] < MIN_POINTS_FOR_MARKER: continue

            center = data['sum_xyz'] / data['count']
            class_name = self.id_to_name.get(class_id, f"ID_{class_id}")
            r, g, b = self.color_cache.get(class_id, (255, 255, 255))
            
            marker_text = Marker()
            marker_text.header.frame_id = FRAME_ID
            marker_text.header.stamp = stamp
            marker_text.ns = "semantic_labels_processed"
            marker_text.id = marker_id_counter
            marker_text.type = Marker.TEXT_VIEW_FACING
            marker_text.action = Marker.ADD
            marker_text.lifetime = MARKER_LIFETIME
            
            marker_text.pose.position.x = float(center[0])
            marker_text.pose.position.y = float(center[1])
            marker_text.pose.position.z = float(center[2]) + MARKER_SCALE_Z 
            
            marker_text.scale.z = MARKER_SCALE_Z  
            marker_text.color.a = 1.0 
            marker_text.color.r = r / 255.0
            marker_text.color.g = g / 255.0
            marker_text.color.b = b / 255.0
            marker_text.text = class_name
            
            marker_array.markers.append(marker_text)
            marker_id_counter += 1
            
        return marker_array
    
    def _read_data_from_files(self):
        if not (os.path.exists(RGB_FILE) and os.path.exists(DEPTH_FILE) and os.path.exists(INFO_FILE)):
            return None, None, None
            
        try:
            cv_rgb = cv2.imread(RGB_FILE)
            cv_depth = np.load(DEPTH_FILE)
            with open(INFO_FILE, 'r') as f:
                info_data = json.load(f)
                K_matrix = info_data['K']
                fx, fy, cx, cy = K_matrix[0], K_matrix[4], K_matrix[2], K_matrix[5]
                K = [fx, fy, cx, cy]
                
            return cv_rgb, cv_depth, K
        except Exception as e:
            rospy.logerr(f"读取或解析临时文件失败: {e}")
            return None, None, None

    def _process_single_batch_with_class_id(self, depth_batch, semantic_batch, u_batch, v_batch, fx, fy, cx, cy, 
                                          points_list, class_centers):
        """处理单个批量的点云数据 - 包含class_id"""
        valid_mask = depth_batch > 0
        if not np.any(valid_mask):
            return 0
        
        valid_depth = depth_batch[valid_mask].astype(np.float32)
        valid_semantic = semantic_batch[valid_mask].astype(np.uint16)  # 保存为uint16
        valid_u = u_batch[valid_mask].astype(np.float32)
        valid_v = v_batch[valid_mask].astype(np.float32)
        
        Z = valid_depth / DEPTH_SCALE
        X = (valid_u - cx) * Z / fx
        Y = (valid_v - cy) * Z / fy
        
        colors = self._generate_colors_batch(valid_semantic)
        rgb_values = self._encode_rgb_to_uint32(colors)
        
        # 构建包含class_id的数据
        # [x, y, z, rgb, class_id]
        batch_data = np.column_stack([X, Y, Z, rgb_values, valid_semantic])
        points_list.append(batch_data)
        
        unique_classes = np.unique(valid_semantic)
        for class_id in unique_classes:
            class_mask = valid_semantic == class_id
            
            if class_id not in class_centers:
                class_centers[class_id] = {
                    'sum_xyz': np.array([
                        np.sum(X[class_mask]),
                        np.sum(Y[class_mask]),
                        np.sum(Z[class_mask])
                    ], dtype=np.float64),
                    'count': np.sum(class_mask)
                }
            else:
                class_centers[class_id]['sum_xyz'] += np.array([
                    np.sum(X[class_mask]),
                    np.sum(Y[class_mask]),
                    np.sum(Z[class_mask])
                ], dtype=np.float64)
                class_centers[class_id]['count'] += np.sum(class_mask)
        
        return len(valid_depth)

    # --- 定时处理函数 ---
    def processing_timer_callback(self, event):
        cv_rgb, cv_depth, K = self._read_data_from_files()
        
        if cv_rgb is None:
            rospy.logwarn(f"定时触发 ({PROCESS_RATE_SEC}s)，但文件数据未就绪或无法读取，跳过处理。")
            return

        rospy.loginfo(f"\n--- 定时触发，开始语义处理 (读取文件系统数据) ---")
        
        start_time = time.time()
        
        try:
            rospy.loginfo("执行 MMSegmentation 推理...")
            semantic_mask = self._get_semantic_mask(cv_rgb)
            
            fx, fy, cx, cy = K
            H, W = semantic_mask.shape
            
            v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            
            depth_flat = cv_depth.flatten()
            semantic_flat = semantic_mask.flatten().astype(np.uint16)  # 转换为uint16
            u_flat = u_coords.flatten()
            v_flat = v_coords.flatten()
            
            total_points = len(depth_flat)
            rospy.loginfo(f"总像素数: {total_points}, 图像尺寸: {W}x{H}")
            
            points_batches = []
            class_centers = {}
            total_valid_points = 0
            
            num_batches = (total_points + VECTORIZED_BATCH_SIZE - 1) // VECTORIZED_BATCH_SIZE
            rospy.loginfo(f"分 {num_batches} 批进行向量化处理，每批最多 {VECTORIZED_BATCH_SIZE} 点")
            
            for i in range(num_batches):
                start_idx = i * VECTORIZED_BATCH_SIZE
                end_idx = min((i + 1) * VECTORIZED_BATCH_SIZE, total_points)
                
                batch_size = self._process_single_batch_with_class_id(
                    depth_flat[start_idx:end_idx],
                    semantic_flat[start_idx:end_idx],
                    u_flat[start_idx:end_idx],
                    v_flat[start_idx:end_idx],
                    fx, fy, cx, cy,
                    points_batches,
                    class_centers
                )
                
                total_valid_points += batch_size
                
                if num_batches > 1 and (i+1) % max(1, num_batches//10) == 0:
                    rospy.loginfo(f"  处理进度: {i+1}/{num_batches} 批, 有效点数: {total_valid_points}")
            
            if total_valid_points == 0:
                rospy.logwarn("未找到有效深度点，跳过本帧")
                return
                
            rospy.loginfo(f"合并 {len(points_batches)} 个批次的数据...")
            
            # 合并所有批次
            if len(points_batches) > 1:
                points_array = np.vstack(points_batches)
            else:
                points_array = points_batches[0]
            
            rospy.loginfo(f"点云数据类型: {points_array.dtype}, 形状: {points_array.shape}")
            rospy.loginfo(f"示例点: x={points_array[0,0]:.3f}, y={points_array[0,1]:.3f}, z={points_array[0,2]:.3f}, "
                         f"rgb={int(points_array[0,3]):#08x}, class_id={int(points_array[0,4])}")
            
            # 构建消息体 - 包含5个字段：x, y, z, rgb, class_id
            pcd_fields = [
                PointField('x', 0, PointField.FLOAT32, 1),      # 偏移0, 4字节
                PointField('y', 4, PointField.FLOAT32, 1),      # 偏移4, 4字节
                PointField('z', 8, PointField.FLOAT32, 1),      # 偏移8, 4字节
                PointField('rgb', 12, PointField.UINT32, 1),    # 偏移12, 4字节
                PointField('class_id', 16, PointField.UINT16, 1) # 偏移16, 2字节
            ]
            # 总大小: 18字节/点
            
            header = rospy.Header()
            header.frame_id = FRAME_ID 
            header.stamp = rospy.Time.now()
            
            # 创建点云消息（包含class_id）
            self.pc_sem_msg = self._create_point_cloud_with_class_id(header, points_array, pcd_fields)
            self.marker_array_msg = self._create_markers(header.stamp, class_centers) 
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            avg_time = np.mean(self.processing_times)
            self.max_processing_time = max(self.max_processing_time, processing_time)
            self.min_processing_time = min(self.min_processing_time, processing_time)
            
            rospy.loginfo(f"处理完成！有效点数: {total_valid_points}")
            rospy.loginfo(f"点云字段: x, y, z, rgb, class_id (共5个字段)")
            rospy.loginfo(f"每个点大小: 18字节 ({total_valid_points}点 = {total_valid_points*18/1024:.1f} KB)")
            rospy.loginfo(f"生成 {len(self.marker_array_msg.markers)} 个语义标签Marker")
            rospy.loginfo(f"处理时间: {processing_time:.3f}s, 平均: {avg_time:.3f}s")
            rospy.loginfo(f"吞吐量: {total_valid_points/processing_time:.0f} 点/秒")

            # 打印类别统计
            self._print_class_statistics(points_array)

            self.data_ready = True

        except Exception as e:
            rospy.logerr(f"定时处理时发生错误: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def _print_class_statistics(self, points_array: np.ndarray):
        """打印类别统计信息"""
        if len(points_array) == 0:
            return
        
        # 提取class_id（第5列）
        class_ids = points_array[:, 4].astype(np.uint16)
        unique_classes = np.unique(class_ids)
        
        table = PrettyTable()
        table.field_names = ["Class ID", "Class Name", "Point Count", "Percentage"]
        table.align["Class Name"] = "l"
        
        total_points = len(points_array)
        
        for class_id in sorted(unique_classes):
            count = np.sum(class_ids == class_id)
            percentage = count / total_points * 100
            class_name = self.id_to_name.get(int(class_id), "UNKNOWN")
            table.add_row([class_id, class_name, count, f"{percentage:.1f}%"])
        
        rospy.loginfo("\n--- 语义类别统计 ---")
        print(table)

    # --- 持续发布函数 ---
    def publish_timer_callback(self, event):
        if not self.data_ready:
            return
            
        current_time = rospy.Time.now()
        
        self.pc_sem_msg.header.stamp = current_time
        
        for marker in self.marker_array_msg.markers:
            marker.header.stamp = current_time
            
        self.pub_pcd_sem.publish(self.pc_sem_msg)
        self.pub_markers.publish(self.marker_array_msg)
        
        rospy.logdebug(f"持续发布带Class ID的点云 ({current_time.to_sec():.2f})...")


if __name__ == '__main__':
    try:
        processor = SemanticProcessorNode()
        rospy.spin() 
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS 中断，语义处理节点退出。")