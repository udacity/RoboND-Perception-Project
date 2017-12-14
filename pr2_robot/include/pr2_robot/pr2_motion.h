/*******************************************************************************
 * Copyright (C) 2017 Udacity Inc.
 *
 * This file is part of Robotic Arm: Perception project for Udacity
 * Robotics nano-degree program
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Harsh Pandya

#ifndef PR2_ROBOT_MOTION_H
#define PR2_ROBOT_MOTION_H

#include <string>
#include <sstream>
#include <ros/ros.h>
#include <std_msgs/Float64.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/RobotTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <geometric_shapes/shape_operations.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf/tf.h>

class PR2Motion
{
public:
  explicit PR2Motion(ros::NodeHandle nh);
  ~PR2Motion();

private:
  ros::NodeHandle nh_;

  const std::string LEFT_PLANNING_GROUP = "left_arm";
  const std::string RIGHT_PLANNING_GROUP = "right_arm";
  const std::string LEFT_GRIPPER_GROUP = "left_gripper";
  const std::string RIGHT_GRIPPER_GROUP = "right_gripper";

  const std::string DROPBOX_MESH_PATH =
    "package://pr2_robot/models/dropbox/meshes/dropbox.dae";

  std::vector<std::string> OBJECT_MESH_PATH_LIST;
  const std::string OBJECT_1_MESH_PATH =
    "package://pr2_robot/models/biscuits/meshes/biscuits.dae";
  const std::string OBJECT_2_MESH_PATH =
    "package://pr2_robot/models/soap/meshes/soap.dae";
  const std::string OBJECT_3_MESH_PATH =
    "package://pr2_robot/models/soap2/meshes/soap2.dae";

  moveit::planning_interface::MoveGroupInterface right_move_group;
  moveit::planning_interface::MoveGroupInterface left_move_group;
  moveit::planning_interface::MoveGroupInterface right_gripper_group;
  moveit::planning_interface::MoveGroupInterface left_gripper_group;


  const robot_state::JointModelGroup *right_joint_model_group;
  const robot_state::JointModelGroup *left_joint_model_group;
  const robot_state::JointModelGroup *right_gripper_joint_model_group;
  const robot_state::JointModelGroup *left_gripper_joint_model_group;

  // Define PlanningSceneInterface object to add and remove collision objects
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  ros::Publisher world_joint_pub;
  /*
   * Functions for gripper actuation
   * close_gripper = 0; open gripper
   *                 = 1; close gripper
   */
  bool OperateLeftGripper(const bool &close_gripper);
  bool OperateRightGripper(const bool &close_gripper);

  bool SetupCollisionObject(const std::string &object_id,
                            const std::string &mesh_path,
                            const geometry_msgs::Pose &object_pose,
                            moveit_msgs::CollisionObject &collision_object);

  tf::Quaternion RPYToQuaternion(float R, float P, float Y);
};

#endif  // PR2_ROBOT_MOTION_H
