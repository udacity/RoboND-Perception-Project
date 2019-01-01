#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
import math
import time
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


TEST_SCENE_NUM = 3

THREAD_IS_EXECUTING = False

DETECTED_POINT_CLOUDS = None

FIRST_RUN = True


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    global THREAD_IS_EXECUTING
    global DETECTED_POINT_CLOUDS
    global DETECTED_OBJECTS
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    # Statistical Outlier Filtering
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    # PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    # PassThrough Filter 2
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Statistical Outlier Filtering
    outlier_filter = cloud_objects.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_objects = outlier_filter.filter()

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(25000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    DETECTED_POINT_CLOUDS = cloud_table

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        # print(feature.shape)
        # print(scaler.mean_.shape)
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!

    if len(set(detected_objects_labels)) == len(detected_objects_labels):
        detected_objects_pub.publish(detected_objects)
        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    if not THREAD_IS_EXECUTING:
        if len(set(detected_objects_labels)) == len(detected_objects_labels):
            detected_objects_pub.publish(detected_objects)
            rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
            time.sleep(10)
            import threading
            t = threading.Thread(target=pr2_mover, args=(detected_objects,))
            t.start()
            THREAD_IS_EXECUTING = True


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    # Initialize variables
    global THREAD_IS_EXECUTING
    global DETECTED_POINT_CLOUDS
    global FIRST_RUN
    labels = []
    centroids = []  # to be list of tuples (x, y, z)
    label_to_centroid_map = {}
    label_to_point_cloud_map = {}
    service_request_dicts = []
    dropbox_to_position_map = {}
    pick_place_args = []

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    for param in dropbox_param:
        dropbox_to_position_map[param['group']] = (param['position'], param['name'])

    for object in object_list:
        label = object.label
        labels.append(label)
        point_cloud = object.cloud
        points_arr = ros_to_pcl(point_cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        centroid = [np.asscalar(p) for p in centroid]
        centroids.append(centroid)
        label_to_centroid_map[label] = centroid
        label_to_point_cloud_map[label] = points_arr

    # Rotate PR2 in place to capture side tables for the collision map
    sleep_time = 30

    rospy.loginfo('Rotating left')
    joint_controller_pub.publish(math.pi/2)
    time.sleep(sleep_time)
    collision_map_pub.publish(pcl_to_ros(DETECTED_POINT_CLOUDS))
    # detected_table_pc.append(DETECTED_POINT_CLOUDS)
    rospy.loginfo('Rotating right')
    joint_controller_pub.publish(-math.pi/2)
    time.sleep(sleep_time*2)
    collision_map_pub.publish(pcl_to_ros(DETECTED_POINT_CLOUDS))
    # detected_table_pc.append(DETECTED_POINT_CLOUDS)
    rospy.loginfo('Rotating back to center')
    joint_controller_pub.publish(0)
    time.sleep(sleep_time)
    collision_map_pub.publish(pcl_to_ros(DETECTED_POINT_CLOUDS))
    # detected_table_pc.append(DETECTED_POINT_CLOUDS)
    rospy.loginfo('Done rotating')

    time.sleep(10)

    # Loop through the pick list
    for i in range(len(object_list_param)):
        object_label = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        if object_label not in label_to_centroid_map.keys():
            rospy.loginfo('Requested objects %s not detected' % object_label)
            print(label_to_centroid_map)
            continue

        # Get the PointCloud for a given object and obtain it's centroid
        centroid = label_to_centroid_map[object_label]

        dropbox_pos, dropbox_name = dropbox_to_position_map[object_group]

        pick_pose = Pose()
        pick_pose.position.x = centroid[0]
        pick_pose.position.y = centroid[1]
        pick_pose.position.z = centroid[2]

        # Create 'place_pose' for the object
        place_pose = Pose()
        place_pose.position.x = dropbox_pos[0]
        place_pose.position.y = dropbox_pos[1]
        place_pose.position.z = dropbox_pos[2]

        # Assign the arm to be used for pick_place
        arm_name = String()
        arm_name.data = dropbox_name
        test_scene_num = Int32()
        test_scene_num.data = TEST_SCENE_NUM
        object_name = String()
        object_name.data = object_label

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        service_request_dicts.append(yaml_dict)
        pick_place_args.append([test_scene_num, arm_name, object_name, pick_pose, place_pose])

    # Wait for 'pick_place_routine' service to come up
    rospy.wait_for_service('pick_place_routine')

    remaining_objects = set(yaml_dict["object_name"] for yaml_dict in service_request_dicts)
    # rospy.loginfo('remaining_objects: %s' % remaining_objects)

    if FIRST_RUN:
        # Output your request parameters into output yaml file
        output_filename = "output_%s.yaml" % TEST_SCENE_NUM
        send_to_yaml(output_filename, service_request_dicts)
        FIRST_RUN = False

    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        # Insert your message variables to be sent as a service request
        for args in pick_place_args:
            test_scene_num, arm_name, object_name, pick_pose, place_pose = args
            remaining_objects -= {object_name.data}

            # TODO Add collision detection for objects
            # collision_objects = cloud_table
            # for label in remaining_objects:
            #     # print(label_to_point_cloud_map[label].shape)
            #     # print(collision_objects.shape)
            #     collision_objects = np.concatenate((collision_objects, label_to_point_cloud_map[label]), axis=0)
            #
            # collision_objects = pcl_to_ros(collision_objects)
            # collision_map_pub.publish(collision_objects)

            rospy.loginfo('picking: %s' % object_name.data)
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print("Response: ", resp.success)

            rospy.loginfo('remaining_objects: %s' % remaining_objects)

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

    THREAD_IS_EXECUTING = False


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    joint_controller_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=1)
    collision_map_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model_%s.sav' % TEST_SCENE_NUM, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    print(encoder.classes_)
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()