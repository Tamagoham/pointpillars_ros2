import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import open3d
OPEN3D_FLAG = True
import torch
import matplotlib
from numpy.lib.recfunctions import unstructured_to_structured
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils
import sensor_msgs_py.point_cloud2 as point_cloud2
from pyquaternion import Quaternion
from sensor_msgs_py.point_cloud2 import PointField

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        if score[i] > 0.4:
            line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
            if ref_labels is None:
                line_set.paint_uniform_color(color)
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])

            vis.add_geometry(line_set)

            # if score is not None:
            #     corners = box3d.get_box_points()
            #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

class PointPillarsNode(Node):
    def __init__(self):
        super().__init__('pointpillars_node')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/top/pointcloud_raw',
            # '/livox/lidar',
            self.listener_callback,
            qos_profile=qos_profile)
        self.publisher = self.create_publisher(PointCloud2, 'detection', 10)
        
        # Load config and model
        cfg_from_yaml_file('/home/ynait0/ros2_ws/src/pointpillars_ros2/pointpillars_ros2/cfgs/nuscenes_models/cbgs_pp_multihead.yaml', cfg)
    
        # Initialize dataset
        self.dataset = DatasetTemplate(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path="/home/ynait0/ros2_ws/src/pointpillars_ros2/"  # データセットのルートパスを設定
        )

        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(cfg.CLASS_NAMES), 
            dataset=self.dataset
        )

        self.model.load_params_from_file(filename='/home/ynait0/ros2_ws/src/pointpillars_ros2/pointpillars_ros2/models/pp_multihead_nds5823_updated.pth', logger=common_utils.create_logger())
        self.model.cuda()
        self.model.eval()

    def listener_callback(self, msg):
        points = self.convert_ros_to_numpy(msg)
        points = self.__convertCloudFormat__(points)
        # points = points.view((np.float32, len(points.dtype.names)))
        scores, dt_box_lidar, types = self.infer(points)

        if scores.size != 0:
            for i in range(scores.size):
                if scores[i] < 0.5:
                    continue
                allow_publishing = True
                if(allow_publishing):
                    print("yaw:", float(dt_box_lidar[i][6]))
                    quat = self.__yawToQuaternion__(float(dt_box_lidar[i][6]))
                    print("orienatation:", quat)
                    print("centerPose:", dt_box_lidar[i][0], dt_box_lidar[i][1], dt_box_lidar[i][2])
                    print("boxSize:", dt_box_lidar[i][3], dt_box_lidar[i][4], dt_box_lidar[i][5])
                    print("type:", types[i])
                    print("score:", scores[i])
                    print("det id:", str(types[i]))

        # self.publisher.publish(detection_msg)

    # def convert_ros_to_numpy(self, msg):
    #     # PointCloud2 message to numpy array conversion
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(np.asarray(list(point_cloud2.read_points(msg))))
    #     return np.asarray(pc.points)

    def __yawToQuaternion__(self, yaw: float) -> Quaternion:
        return Quaternion(axis=[0, 0, 1], radians=yaw)

    def convert_ros_to_numpy(self, msg):
        # PointCloud2 message to numpy array conversion
        # print(msg)
        dtype_list = fields_to_dtype(msg.fields, msg.point_step)
        # print(dtype_list)
        cloud_arr = np.frombuffer(msg.data, dtype_list)

        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (
                fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
        
        if msg.height == 1:
            return np.reshape(cloud_arr, (msg.width,))
        else:
            return np.reshape(cloud_arr, (msg.height, msg.width))

    
    def __convertCloudFormat__(self, cloud_array, remove_nans=True, dtype=np.float32):
        '''
        '''
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        
        points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']
        if 'intensity' in cloud_array.dtype.names:
            points[...,3] = cloud_array['intensity']
        return points

    def infer(self, points):
        if len(points) == 0:
            return 0, 0, 0
        # PointPillars inference
        # points.reshape(-1, 4) 
        input_dict = {
            'points': points,
        }
        # data_dict = DatasetTemplate.prepare_data(data_dict=input_dict)
        data_dict = self.dataset.prepare_data(data_dict=input_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            torch.cuda.synchronize()
            # print(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            torch.cuda.synchronize()

            boxes_lidar = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
            print(pred_dicts[0]['pred_labels'])

            draw_scenes(
                points=points, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            # print(boxes_lidar)
            # print(scores)
            # print(labels)

        return scores, boxes_lidar, labels

    def convert_numpy_to_ros(self, predictions):
        pass
        # print(predictions)

        # # Convert numpy array to PointCloud2 message
        # header = Header()
        # header.stamp = self.get_clock().now().to_msg()
        # header.frame_id = 'map'
        # return point_cloud2.create_cloud(header, self.pcl2_fields, predictions)

def main(args=None):
    rclpy.init(args=args)
    pointpillars_node = PointPillarsNode()
    rclpy.spin(pointpillars_node)
    pointpillars_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



