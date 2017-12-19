/*******************************************************************************
 * Copyright (C) 2017 Udacity Inc.
 *
 * This file is part of Robotic Arm: Perception project for Udacity
 * Robotics nano-degree program
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Harsh Pandya

#include<pr2_robot/pr2_motion.h>

PR2Motion::PR2Motion(ros::NodeHandle nh)
  : nh_(nh),
    right_move_group(RIGHT_PLANNING_GROUP),
    left_move_group(LEFT_PLANNING_GROUP),
    right_gripper_group(RIGHT_GRIPPER_GROUP),
    left_gripper_group(LEFT_GRIPPER_GROUP)
{
  // Pointer to JointModelGroup for improved performance.
  right_joint_model_group =
    right_move_group.getCurrentState()->getJointModelGroup(RIGHT_PLANNING_GROUP);
  left_joint_model_group =
    left_move_group.getCurrentState()->getJointModelGroup(LEFT_PLANNING_GROUP);
  right_gripper_joint_model_group =
    right_gripper_group.getCurrentState()->getJointModelGroup(RIGHT_GRIPPER_GROUP);
  left_gripper_joint_model_group =
    left_gripper_group.getCurrentState()->getJointModelGroup(LEFT_GRIPPER_GROUP);

  world_joint_pub = nh.advertise<std_msgs::Float64>("/pr2/world_joint_controller/command", 1);

  /*
   * rviz visualization:
   * Setup MoveItVisualTools for visualizing collision objects, robot,
   * and trajectories in Rviz
   */
  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_ptr;
  visual_tools_ptr.reset(new moveit_visual_tools::MoveItVisualTools("world"));
  visual_tools_ptr->deleteAllMarkers();

  // Load RemoteControl for step-by-step progression
  visual_tools_ptr->loadRemoteControl();

  // Create text marker for displaying current state
  Eigen::Affine3d text_pose = Eigen::Affine3d::Identity();
  text_pose.translation().z() = 1.5;
  visual_tools_ptr->publishText(text_pose, "Welcome to Advance Pick and Place project",
                           rviz_visual_tools::WHITE, rviz_visual_tools::XXXLARGE);

  // Publish messages to rviz
  visual_tools_ptr->trigger();
  visual_tools_ptr->prompt("next step");

  /*
   * Collision Objects:
   * Create an object list and populate it with shelf and bin objects
   * Then insert objects in scene for collision avoidance and interaction
   */
  std::vector<moveit_msgs::CollisionObject> collision_object_list;
  std::vector<std::string> object_ids;
  moveit_msgs::CollisionObject left_dropbox_collision_object,
                               right_dropbox_collision_object;

  // Define pose for the objects (specified relative to base_footprint)
  geometry_msgs::Pose left_mesh_pose, right_mesh_pose;

  left_mesh_pose.position.x = 0;
  left_mesh_pose.position.y = 0.7;
  left_mesh_pose.position.z = 0.605;
  left_mesh_pose.orientation.w = 0.707;
  left_mesh_pose.orientation.x = 0;
  left_mesh_pose.orientation.y = 0;
  left_mesh_pose.orientation.z = 0.707;

  right_mesh_pose.position.x = 0;
  right_mesh_pose.position.y = -0.71;
  right_mesh_pose.position.z = 0.605;
  right_mesh_pose.orientation.w = 0.707;
  right_mesh_pose.orientation.x = 0;
  right_mesh_pose.orientation.y = 0;
  right_mesh_pose.orientation.z = 0.707;

  SetupCollisionObject("left_dropbox", DROPBOX_MESH_PATH, left_mesh_pose,
                       left_dropbox_collision_object);
  SetupCollisionObject("right_dropbox", DROPBOX_MESH_PATH, right_mesh_pose,
                       right_dropbox_collision_object);

  collision_object_list.push_back(left_dropbox_collision_object);
  collision_object_list.push_back(right_dropbox_collision_object);

  // Add the object list to the world scene
  planning_scene_interface.addCollisionObjects(collision_object_list);
  ROS_INFO("Added object list to the world");

  // Allow MoveGroup to add the collision objects in the world
  ros::Duration(1.0).sleep();
  //visual_tools_ptr->prompt("next step");

  // We can print the name of the reference frame for this robot.
  ROS_INFO_NAMED("tutorial", "Reference frame: %s", right_move_group.getPlanningFrame().c_str());

  // We can also print the name of the end-effector link for this group.
  ROS_INFO_NAMED("tutorial", "End effector link: %s", right_move_group.getEndEffectorLink().c_str());
  //Tuck arms
  right_move_group.setNamedTarget("RIGHT_ARM_INITIAL_POSE");
  left_move_group.setNamedTarget("LEFT_ARM_INITIAL_POSE");

  moveit::planning_interface::MoveGroupInterface::Plan right_arm_plan, left_arm_plan;

  bool right_success = right_move_group.move();
  bool left_success = left_move_group.move();

  ROS_INFO_NAMED("tutorial", "Visualizing plan 1 (pose goal) %s", (right_success & left_success) ? "" : "FAILED");

  visual_tools_ptr->prompt("next step");

  //Rotate in place to capture collision map from the sides
  std_msgs::Float64 world_joint_value;

  world_joint_value.data = -1.57;
  world_joint_pub.publish(world_joint_value);
  ros::Duration(1.0).sleep();
  visual_tools_ptr->prompt("next step");

  world_joint_value.data = 1.57;
  world_joint_pub.publish(world_joint_value);
  ros::Duration(1.0).sleep();
  visual_tools_ptr->prompt("next step");

  world_joint_value.data = 0;
  world_joint_pub.publish(world_joint_value);
  ros::Duration(1.0).sleep();
  visual_tools_ptr->prompt("next step");

  //Target object pick pose
  std::vector<geometry_msgs::Pose> pose_list, drop_list, mesh_pose_list;
  geometry_msgs::Pose target_pose, drop_pose, target_mesh_pose;

  drop_pose.orientation.w = 1.0;
  drop_pose.position.x = -0.03;
  drop_pose.position.y = -0.8;
  drop_pose.position.z = 1.0;
  drop_list.push_back(drop_pose);

  drop_pose.orientation.w = 1.0;
  drop_pose.position.x = -0.03;
  drop_pose.position.y = -0.65;
  drop_pose.position.z = 1.0;
  drop_list.push_back(drop_pose);

  drop_pose.orientation.w = 1.0;
  drop_pose.position.x = -0.1;
  drop_pose.position.y = -0.8;
  drop_pose.position.z = 1.0;
  drop_list.push_back(drop_pose);

  tf::Quaternion qt = RPYToQuaternion(1.57, 1.57, 0);
  target_pose.orientation.w = qt.getW();
  target_pose.orientation.x = qt.getX();
  target_pose.orientation.y = qt.getY();
  target_pose.orientation.z = qt.getZ();
  target_pose.position.x = 0.56;
  target_pose.position.y = -0.24;
  target_pose.position.z = 0.97;
  pose_list.push_back(target_pose);

  qt = RPYToQuaternion(-0.75, 1.57, 0);
  target_pose.orientation.w = qt.getW();
  target_pose.orientation.x = qt.getX();
  target_pose.orientation.y = qt.getY();
  target_pose.orientation.z = qt.getZ();
  target_pose.position.x = 0.57;
  target_pose.position.y = -0.02;
  target_pose.position.z = 0.93;
  pose_list.push_back(target_pose);

  qt = RPYToQuaternion(0.85, 1.57, 0);
  target_pose.orientation.w = qt.getW();
  target_pose.orientation.x = qt.getX();
  target_pose.orientation.y = qt.getY();
  target_pose.orientation.z = qt.getZ();
  target_pose.position.x = 0.46;
  target_pose.position.y = 0.23;
  target_pose.position.z = 0.95;
  pose_list.push_back(target_pose);

  //Mesh poses
  OBJECT_MESH_PATH_LIST.push_back(OBJECT_1_MESH_PATH);
  OBJECT_MESH_PATH_LIST.push_back(OBJECT_2_MESH_PATH);
  OBJECT_MESH_PATH_LIST.push_back(OBJECT_3_MESH_PATH);

  qt = RPYToQuaternion(0, 0, -0.79);
  target_mesh_pose.position.x = 0.54;
  target_mesh_pose.position.y = -0.24;
  target_mesh_pose.position.z = 0.63;
  target_mesh_pose.orientation.w = qt.getW();
  target_mesh_pose.orientation.x = qt.getX();
  target_mesh_pose.orientation.y = qt.getY();
  target_mesh_pose.orientation.z = qt.getZ();
  mesh_pose_list.push_back(target_mesh_pose);

  qt = RPYToQuaternion(0, 0, -1.35);
  target_mesh_pose.position.x = 0.56;
  target_mesh_pose.position.y = 0;
  target_mesh_pose.position.z = 0.615;
  target_mesh_pose.orientation.w = qt.getW();
  target_mesh_pose.orientation.x = qt.getX();
  target_mesh_pose.orientation.y = qt.getY();
  target_mesh_pose.orientation.z = qt.getZ();
  mesh_pose_list.push_back(target_mesh_pose);

  qt = RPYToQuaternion(0, 0, 0);
  target_mesh_pose.position.x = 0.46;
  target_mesh_pose.position.y = 0.22;
  target_mesh_pose.position.z = 0.61;
  target_mesh_pose.orientation.w = qt.getW();
  target_mesh_pose.orientation.x = qt.getX();
  target_mesh_pose.orientation.y = qt.getY();
  target_mesh_pose.orientation.z = qt.getZ();
  mesh_pose_list.push_back(target_mesh_pose);

  //Add object to the scene
  std::vector<moveit_msgs::CollisionObject> target_object_list;
  for(size_t i=0; i<pose_list.size();++i)
  {
    moveit_msgs::CollisionObject target_collision_object;
    std::string target_id;
    target_id = "target"+ std::to_string (i);

    SetupCollisionObject(target_id, OBJECT_MESH_PATH_LIST[i], mesh_pose_list[i], target_collision_object);

    target_object_list.push_back(target_collision_object);
  }

  // Add the object list to the world scene
  planning_scene_interface.addCollisionObjects(target_object_list);
  ROS_INFO("Added Target to the world");

  // Allow MoveGroup to add the collision objects in the world
  ros::Duration(5.0).sleep();


  ROS_INFO_STREAM(pose_list[0]);

  for(size_t i=0; i<pose_list.size();++i)
  {
    // set starting pose
    right_move_group.setStartStateToCurrentState();
    left_move_group.setStartStateToCurrentState();

    // set target pose
    right_move_group.setPoseTarget(pose_list[i]);
    right_success = right_move_group.plan(right_arm_plan);
    ROS_INFO("Visualizing plan to target: %s",
             right_success ? "SUCCEEDED" : "FAILED");

    // visualize the plan in Rviz.
    ROS_INFO("Visualizing plan 1 as trajectory line");
    visual_tools_ptr->deleteAllMarkers();
    visual_tools_ptr->publishAxisLabeled(pose_list[i], "reach_pose");
    visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
    visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
    visual_tools_ptr->trigger();
    visual_tools_ptr->prompt("next step");

    right_move_group.execute(right_arm_plan);

    //Reach movement
    right_move_group.setStartStateToCurrentState();
    pose_list[i].position.z = pose_list[i].position.z-0.07;
    right_move_group.setPoseTarget(pose_list[i]);
    right_success = right_move_group.plan(right_arm_plan);
    ROS_INFO("Visualizing plan to target: %s",
             right_success ? "SUCCEEDED" : "FAILED");
    // We can also visualize the plan as a line with markers in Rviz.
    visual_tools_ptr->deleteAllMarkers();
    visual_tools_ptr->publishAxisLabeled(pose_list[i], "pick_pose");
    visual_tools_ptr->publishText(text_pose, "Pick Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
    visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
    visual_tools_ptr->trigger();
    visual_tools_ptr->prompt("next step");

    right_move_group.execute(right_arm_plan);

    //Remove object from the scene
    object_ids.push_back(target_object_list[i].id);
    planning_scene_interface.removeCollisionObjects(object_ids);

    //Close Gripper
    OperateRightGripper(true);
    ros::Duration(3.0).sleep();

    //Reach movement
    right_move_group.setStartStateToCurrentState();
    pose_list[i].position.z = pose_list[i].position.z+0.12;
    right_move_group.setPoseTarget(pose_list[i]);
    right_success = right_move_group.plan(right_arm_plan);
    ROS_INFO("Visualizing plan to target: %s",
             right_success ? "SUCCEEDED" : "FAILED");
    // visualize the plan in Rviz.
    visual_tools_ptr->deleteAllMarkers();
    visual_tools_ptr->publishAxisLabeled(pose_list[i], "reach_pose");
    visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
    visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
    visual_tools_ptr->trigger();
    visual_tools_ptr->prompt("next step");

    right_move_group.execute(right_arm_plan);

    //drop the ball
    right_move_group.setStartStateToCurrentState();
    right_move_group.setPoseTarget(drop_list[i]);
    right_success = right_move_group.plan(right_arm_plan);
    ROS_INFO("Visualizing plan to target: %s",
             right_success ? "SUCCEEDED" : "FAILED");

    visual_tools_ptr->deleteAllMarkers();
    visual_tools_ptr->publishAxisLabeled(drop_list[i], "drop_pose");
    visual_tools_ptr->publishText(text_pose, "Drop Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
    visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
    visual_tools_ptr->trigger();
    visual_tools_ptr->prompt("next step");

    right_move_group.execute(right_arm_plan);

    //Open Gripper
    OperateRightGripper(false);
  }

  visual_tools_ptr->deleteAllMarkers();

  // Create text marker for displaying current state
  text_pose.translation().z() = 2.0;
  visual_tools_ptr->publishText(text_pose, "End.",
                           rviz_visual_tools::WHITE, rviz_visual_tools::XXXLARGE);

  // Publish messages to rviz
  visual_tools_ptr->trigger();

  // set starting pose
  right_move_group.setStartStateToCurrentState();
  left_move_group.setStartStateToCurrentState();

  right_move_group.setNamedTarget("RIGHT_ARM_INITIAL_POSE");
  left_move_group.setNamedTarget("LEFT_ARM_INITIAL_POSE");
  right_success = right_move_group.move();
  left_success = left_move_group.move();
}


bool PR2Motion::SetupCollisionObject(const std::string &object_id,
    const std::string &mesh_path,
    const geometry_msgs::Pose &object_pose,
    moveit_msgs::CollisionObject &collision_object)
{
  collision_object.header.frame_id = right_move_group.getPlanningFrame();
  collision_object.id = object_id;

  shapes::Mesh* m = shapes::createMeshFromResource(mesh_path);

  ROS_DEBUG_STREAM(object_id << " mesh loaded");

  shape_msgs::Mesh object_mesh;
  shapes::ShapeMsg object_mesh_msg;
  shapes::constructMsgFromShape(m, object_mesh_msg);
  object_mesh = boost::get<shape_msgs::Mesh>(object_mesh_msg);
  collision_object.meshes.resize(1);
  collision_object.mesh_poses.resize(1);
  collision_object.meshes[0] = object_mesh;

  collision_object.mesh_poses[0].position = object_pose.position;
  collision_object.mesh_poses[0].orientation = object_pose.orientation;

  collision_object.meshes.push_back(object_mesh);
  collision_object.mesh_poses.push_back(collision_object.mesh_poses[0]);
  collision_object.operation = collision_object.ADD;
}

bool PR2Motion::OperateRightGripper(const bool &close_gripper)
{
  // RobotState contains the current position/velocity/acceleration data
  moveit::core::RobotStatePtr gripper_current_state =
    right_gripper_group.getCurrentState();

  // Next get the current set of joint values for the group.
  std::vector<double> gripper_joint_positions;
  gripper_current_state->copyJointGroupPositions(right_gripper_joint_model_group,
      gripper_joint_positions);

  ROS_DEBUG("No. of joints in eef_group: %zd", gripper_joint_positions.size());

  // Set finger joint values
  if (close_gripper)
  {
    gripper_joint_positions[0] = 0.04;
    gripper_joint_positions[1] = 0.04;
  }
  else
  {
    gripper_joint_positions[0] = 0.0;
    gripper_joint_positions[1] = 0.0;
  }

  right_gripper_group.setJointValueTarget(gripper_joint_positions);
  ros::Duration(1.5).sleep();

  bool success = right_gripper_group.move();
  return success;
}

bool PR2Motion::OperateLeftGripper(const bool &close_gripper)
{
  // RobotState contains the current position/velocity/acceleration data
  moveit::core::RobotStatePtr gripper_current_state =
    left_gripper_group.getCurrentState();

  // Next get the current set of joint values for the group.
  std::vector<double> gripper_joint_positions;
  gripper_current_state->copyJointGroupPositions(left_gripper_joint_model_group,
      gripper_joint_positions);

  ROS_DEBUG("No. of joints in eef_group: %zd", gripper_joint_positions.size());

  // Set finger joint values
  if (close_gripper)
  {
    gripper_joint_positions[0] = 0.03;
    gripper_joint_positions[1] = 0.03;
  }
  else
  {
    gripper_joint_positions[0] = 0.0;
    gripper_joint_positions[1] = 0.0;
  }

  left_gripper_group.setJointValueTarget(gripper_joint_positions);
  ros::Duration(1.5).sleep();

  bool success = left_gripper_group.move();
  return success;
}

tf::Quaternion PR2Motion::RPYToQuaternion(float R, float P, float Y)
{
  tf::Matrix3x3 mat;
  mat.setEulerYPR(Y,P,R);

  tf::Quaternion quat;
  mat.getRotation(quat);

  return quat;
}

PR2Motion::~PR2Motion(){}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "pr2_motion");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(4);
  spinner.start();
  PR2Motion pr2_motion(nh);
  return 0;
}
