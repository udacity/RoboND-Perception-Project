#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import String
from pr2_robot.srv import *

def get_grasp_handler(req):
    object_name = req.object_name.data
    test_num = req.test_scene_num.data-1
    grasp_pose = Pose()

    grasp_list = rospy.get_param('/grasp_list')

    grasp_pose.position.x = grasp_list[test_num][object_name]['position']['x']
    grasp_pose.position.y = grasp_list[test_num][object_name]['position']['y']
    grasp_pose.position.z = grasp_list[test_num][object_name]['position']['z']

    grasp_pose.orientation.w = grasp_list[test_num][object_name]['orientation']['w']
    grasp_pose.orientation.x = grasp_list[test_num][object_name]['orientation']['x']
    grasp_pose.orientation.y = grasp_list[test_num][object_name]['orientation']['y']
    grasp_pose.orientation.z = grasp_list[test_num][object_name]['orientation']['z']

    return GraspResponse(grasp_pose)

def grasp_server():
    rospy.init_node('grasp_server')
    s = rospy.Service('get_grasp', Grasp, get_grasp_handler)
    print "Ready to retrieve grasps"
    rospy.spin()

if __name__ == '__main__':
    try:
        grasp_server()
    except rospy.ROSInterruptException:
        pass
