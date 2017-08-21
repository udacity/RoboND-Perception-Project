#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
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
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import re
import xml.etree.ElementTree
import logging

class perceptionState():
    def __init__(self, *args, **kwargs):
        self.test_scene_num = 0
        self.object_to_pick = 0
        self.rotated = False
        self.collsion_cloud = None
        self.body_angle = [-1.57,1.57, 1.57, 0, 0]
        self.body_angle_index = 0
    def scene_check(self, launch_file): #../launch/pick_place_project.launch
        e = xml.etree.ElementTree.parse(launch_file).getroot()

        world_num_regex = re.compile(r'worlds/(.*).world')
        scene_num_regex = re.compile(r'\w+(\d+)')
        challange_regex = re.compile(r'(\w+)')

        for atype in e.findall('.//include/arg'):
            scene_num = world_num_regex.search(atype.get('value'))
            if scene_num is not None:
                num = scene_num_regex.search(scene_num.group(1))
                if num is not None:
                    self.test_scene_num = num.group(1)
                else:
                    challange = challange_regex.search(scene_num.group(1))
                    self.test_scene_num = challange.group(1)
                break
        logging.debug('Test_scene_num: %s' % self.test_scene_num)
    def set_collision_cloud(self, cloud):
        self.collsion_cloud =cloud

    def get_collision_cloud(self):
        return self.collsion_cloud
# Helper function to existence check

def existanceCheck(objectName, object_list_param):
    for index, object in enumerate(object_list_param):
        if object["name"] == objectName:
            return True, index

    return False, -1

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        outfile.write(yaml.dump(data_dict, default_flow_style=False))

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    # Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object:
    outlier_filter = pcl_data.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(10)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter z
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.6
    axis_max = 0.9
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # PassThrough Filter x
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.00075
    seg.set_distance_threshold(max_distance)

    # Extract inliers and outliers
    inliers, coefficients = seg.segment()
    extracted_inliers = cloud_filtered.extract(inliers,negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # Publish a point cloud to `/pr2/3D_map/points`.
    # Telling the robot where objects are in the environment in order to avoid collisions.
    # check if the collision map was maded by checking perceptionState.rotated == true
    collision_map_3d = pcl_to_ros(cloud_filtered)
    collision_map_pub.publish(collision_map_3d)

    # 10 second to finish 1000/5 = 20
    rate = rospy.Rate(0.08)

    if perception_state.rotated == True:

        # Euclidean Clustering
        white_cloud = XYZRGB_to_XYZ(extracted_outliers)
        tree = white_cloud.make_kdtree()
        ec = white_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.005)
        ec.set_MinClusterSize(80)
        ec.set_MaxClusterSize(2500)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        # Create Cluster-Mask Point Cloud to visualize each cluster separately
        cluster_color = get_color_list(len(cluster_indices))

        color_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([white_cloud[indice][0],
                                                white_cloud[indice][1],
                                                white_cloud[indice][2],
                                                rgb_to_float(cluster_color[j])])

        cluster_out = pcl.PointCloud_PointXYZRGB()
        cluster_out.from_list(color_cluster_point_list)

        # Convert PCL data to ROS messages
        ros_cloud_objects = pcl_to_ros(extracted_outliers)
        ros_cloud_table = pcl_to_ros(extracted_inliers)
        ros_cluster_cloud = pcl_to_ros(cluster_out)

        # Publish ROS messages
        pcl_objects_pub.publish(ros_cloud_objects)
        pcl_table_pub.publish(ros_cloud_table)
        pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs:

        detected_objects_labels = []
        detected_objects = []

        for index, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster
            pcl_cluster = extracted_outliers.extract(pts_list)
            # convert the cluster from pcl to ROS using helper function
            ros_cluster = pcl_to_ros(pcl_cluster)
            # Compute the associated feature vector

            # Extract histogram features
            chists = compute_color_histograms(ros_cluster, using_hsv=True)
            normals = get_normals(ros_cluster)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            #detected_objects_labels.append([feature, index])

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
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

        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
        logging.debug('Detected %s' % (len(detected_objects_labels)))

        # Publish the list of detected objects
        # This is the output you'll need to complete the upcoming project!
        detected_objects_pub.publish(detected_objects)

        # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
        # Could add some logic to determine whether or not your object detections are robust
        # before calling pr2_mover()
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
            logging.warning('ROS interrupt: (%s)' % (rospy.ROSInterruptException))
    else:
        # TODO: Rotate PR2 in place to capture side tables for the collision map
        #But just rotation does not do any thing here
        #This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
        pr2_joint_pub.publish(perception_state.body_angle[perception_state.body_angle_index])
        rate.sleep()
        perception_state.body_angle_index += 1
        if perception_state.body_angle_index == 5:
            perception_state.rotated = True

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    object_list_param = []
    dropbox_param = []
    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    test_scene_num = Int32()
    test_scene_num.data = int(perception_state.test_scene_num) -1

    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()



    yaml_dict_list = []
    labels = []
    pick_centroids = [] # to be detected object list of tuples (x, y, z)
    place_centroids = [] # to be placing bin list of tuples (x, y, z)

    # Loop through the pick list
    for index in range(0,len(object_list)):
        existence, i = existanceCheck(object_list[index].label, object_list_param)
        if existence:
            object_name.data = str(object_list[index].label)
            object_group = object_list_param[i]['group']

            # Get the PointCloud for a given object and obtain it's centroid
            labels.append(object_list[index].label)
            points_arr = ros_to_pcl(object_list[index].cloud).to_array()
            pick_centroid = np.mean(points_arr, axis=0)[:3]
            pick_pose.position.x = float(pick_centroid[0])
            pick_pose.position.y = float(pick_centroid[1])
            pick_pose.position.z = float(pick_centroid[2])
            pick_pose.orientation.x = 0.0
            pick_pose.orientation.y = 0.0
            pick_pose.orientation.z = 0.0
            pick_pose.orientation.w = 0.0
            pick_centroids.append(pick_centroid)

            # Create 'place_pose' for the object
            for j in range(0,len(dropbox_param)):
                if object_group == dropbox_param[j]["group"]:
                    place_centroid = dropbox_param[j]["position"]
                    place_pose.position.x = float(place_centroid[0])
                    place_pose.position.y = float(place_centroid[1])
                    place_pose.position.z = float(place_centroid[2])
                    place_pose.orientation.x = 0.0
                    place_pose.orientation.y = 0.0
                    place_pose.orientation.z = 0.0
                    place_pose.orientation.w = 0.0
                    place_centroids.append(place_centroid)

                    # Assign the arm to be used for pick_place
                    arm_name.data = dropbox_param[j]["name"]

            # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

            yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            yaml_dict_list.append(yaml_dict)
            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')

            try:
                pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

                # Insert your message variables to be sent as a service request
                logging.debug('Sending: (%s, %s, %s, %s, %s)' % (str(int(test_scene_num.data)+1), str(object_name), str(arm_name), str(pick_pose), str(place_pose)))

                resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

                print ("Response: ",resp.success)
                logging.debug('Response: (%s)' % (str(resp.success)))
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
                logging.warning('Service call failed: (%s)' % (str(e)))
        else:

            print "error in recognition: %s not inclusive in the object list" %object_list[index].label
            logging.debug('error in recognition: %s not inclusive in the object list' % object_list[index].label)
    # Output your request parameters into output yaml file
    yaml_filename = "output.yml"
    # make it to reflect the test case
    send_to_yaml(yaml_filename, yaml_dict_list)


if __name__ == '__main__':
    logging.basicConfig(filename='debuggingInfo.txt',level=logging.DEBUG, format=' %(asctime)s -- %(message)s')
    #logging.disable(logging.CRITICAL)
    logging.debug('Start of program')
    #perception state class initialisation
    perception_state = perceptionState()
    # Reading the pick_place_project.launch xml file and regex line 13 <arg name="world_name" value="$(find pr2_robot)/worlds/test1.world"/>
    # call xml reader to parse test_scene_num
    #checking which scene will be used, checking only once in this python script
    #before going into the service subroutine updating the cls state
    perception_state.scene_check('../launch/pick_place_project.launch')
    logging.debug('Initialisation started')
    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)


    # Create Subscribers
    #pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    # here you need to create two publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    collision_map_pub = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size=1)
    pr2_joint_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=1)
    #pr2_joint_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=1)
    #  Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
