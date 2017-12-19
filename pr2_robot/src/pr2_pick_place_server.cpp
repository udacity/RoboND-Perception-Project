/*******************************************************************************
 * Copyright (C) 2017 Udacity Inc.
 *
 * This file is part of Robotic Arm: Perception project for Udacity
 * Robotics nano-degree program
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Harsh Pandya

#include<pr2_robot/pr2_pick_place_server.h>

PR2PickPlace::PR2PickPlace(ros::NodeHandle nh)
  : nh_(nh),
    right_move_group(RIGHT_PLANNING_GROUP),
    left_move_group(LEFT_PLANNING_GROUP),
    right_gripper_group(RIGHT_GRIPPER_GROUP),
    left_gripper_group(LEFT_GRIPPER_GROUP)
{
  client = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
  grasp_client = nh.serviceClient<pr2_robot::Grasp>("/get_grasp");

  // Pointer to JointModelGroup for improved performance.
  right_joint_model_group =
    right_move_group.getCurrentState()->getJointModelGroup(RIGHT_PLANNING_GROUP);
  left_joint_model_group =
    left_move_group.getCurrentState()->getJointModelGroup(LEFT_PLANNING_GROUP);
  right_gripper_joint_model_group =
    right_gripper_group.getCurrentState()->getJointModelGroup(RIGHT_GRIPPER_GROUP);
  left_gripper_joint_model_group =
    left_gripper_group.getCurrentState()->getJointModelGroup(LEFT_GRIPPER_GROUP);

  visual_tools_ptr.reset(new moveit_visual_tools::MoveItVisualTools("world"));
  visual_tools_ptr->deleteAllMarkers();
  visual_tools_ptr->loadRemoteControl();

  // Create text marker for displaying current state
  Eigen::Affine3d text_pose = Eigen::Affine3d::Identity();
  text_pose.translation().z() = 1.6;
  visual_tools_ptr->publishText(text_pose, "Welcome to Advance Pick and Place project",
                           rviz_visual_tools::WHITE, rviz_visual_tools::XXXLARGE);

  // Publish messages to rviz
  visual_tools_ptr->trigger();

  /*
   * Collision Objects:
   * Create an object list and populate it with dropbox objects
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
  ros::Duration(1.0).sleep();

  //Raise arms
  right_move_group.setNamedTarget("RIGHT_ARM_INITIAL_POSE");
  left_move_group.setNamedTarget("LEFT_ARM_INITIAL_POSE");

  right_success = right_move_group.move();
  left_success = left_move_group.move();

}

bool PR2PickPlace::Routine(pr2_robot::PickPlace::Request &req,
                    pr2_robot::PickPlace::Response &res)
{

  // test
  visual_tools_ptr->deleteAllMarkers();

  // Create text marker for displaying current state
  Eigen::Affine3d text_pose = Eigen::Affine3d::Identity();
  text_pose.translation().z() = 1.6;
  visual_tools_ptr->publishText(text_pose, "New request received",
                           rviz_visual_tools::WHITE, rviz_visual_tools::XXXLARGE);
  visual_tools_ptr->trigger();

  //Is pick_pose close to actual object pose
  gazebo_msgs::GetModelState srv;
  pr2_robot::Grasp grasp_srv;
  geometry_msgs::Pose act_obj_pose, place_pose, grasp_pose;

  place_pose = req.place_pose;

  grasp_srv.request.object_name = req.object_name;
  grasp_srv.request.test_scene_num = req.test_scene_num;

  if (grasp_client.call(grasp_srv))
  {
    grasp_pose = grasp_srv.response.grasp_pose;
    ROS_INFO_STREAM("grasp_pose: "<<grasp_pose);
  }
  else
  {
    ROS_ERROR("Failed to call get_grasp service ");
    return 1;
  }

  srv.request.model_name = req.object_name.data;
  srv.request.relative_entity_name = "world";

  if (client.call(srv))
  {
    act_obj_pose = srv.response.pose;
  }
  else
  {
    ROS_ERROR("Failed to call service GetModelState");
    return 1;
  }

  bool success = IsPickPoseWithinLimits(req.pick_pose, act_obj_pose);

  if(!success)
  {
    ROS_INFO_STREAM("Incorrect pick_pose for: "<<req.object_name.data);
    res.success = false;
  }

  else
  {

    // Pick Pose is within limits, next spawn the collision object
    geometry_msgs::Pose target_mesh_pose;
    std::vector<moveit_msgs::CollisionObject> target_object_list;
    std::vector<std::string> object_ids;
    moveit_msgs::CollisionObject target_collision_object;

    std::string TARGET_MESH_PATH = OBJECT_MESH_PATH + req.object_name.data + "/meshes/" + req.object_name.data + ".dae";

    SetupCollisionObject(req.object_name.data, TARGET_MESH_PATH, act_obj_pose, target_collision_object);

    target_object_list.push_back(target_collision_object);
    // Add the object list to the world scene
    planning_scene_interface.addCollisionObjects(target_object_list);
    ROS_INFO("Added Target to the world");
    ros::Duration(1.0).sleep();

    // Plan arm motion
    // set starting pose
    right_move_group.setStartStateToCurrentState();
    left_move_group.setStartStateToCurrentState();

    // set target pose

    if(req.arm_name.data == "right")
    {
      right_move_group.setPoseTarget(grasp_pose);

      right_success = right_move_group.plan(right_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               right_success ? "SUCCEEDED" : "FAILED");

      // We can also visualize the plan as a line with markers in Rviz.
      ROS_INFO("Visualizing plan 1 as trajectory line");
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "reach_pose");
      visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("Click Next");

      right_move_group.execute(right_arm_plan);

      //Reach movement
      right_move_group.setStartStateToCurrentState();
      grasp_pose.position.z = grasp_pose.position.z-0.07;
      right_move_group.setPoseTarget(grasp_pose);
      right_success = right_move_group.plan(right_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               right_success ? "SUCCEEDED" : "FAILED");

      // We can also visualize the plan as a line with markers in Rviz.
      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "pick_pose");
      visual_tools_ptr->publishText(text_pose, "Pick Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("Click Next");

      right_move_group.execute(right_arm_plan);

      //Remove object from the scene
      object_ids.push_back(req.object_name.data);
      planning_scene_interface.removeCollisionObjects(object_ids);

      //Close Gripper
      OperateRightGripper(true);
      ros::Duration(3.0).sleep();

      //Reach movement
      right_move_group.setStartStateToCurrentState();
      grasp_pose.position.z = grasp_pose.position.z+0.12;
      right_move_group.setPoseTarget(grasp_pose);
      right_success = right_move_group.plan(right_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               right_success ? "SUCCEEDED" : "FAILED");
      // visualize the plan in Rviz.
      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "reach_pose");
      visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("Click Next");

      right_move_group.execute(right_arm_plan);

      //drop the ball
      right_move_group.setStartStateToCurrentState();
      place_pose.position.z = place_pose.position.z+0.4;

      ROS_INFO_STREAM(place_pose);
      //visual_tools_ptr->prompt("Click Next");

      right_move_group.setPoseTarget(place_pose);
      right_success = right_move_group.plan(right_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               right_success ? "SUCCEEDED" : "FAILED");

      visual_tools_ptr->publishAxisLabeled(place_pose, "drop_pose");
      visual_tools_ptr->publishText(text_pose, "Drop Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(right_arm_plan.trajectory_, right_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("Click Next");
      right_move_group.execute(right_arm_plan);


      //Open Gripper
      OperateRightGripper(false);
      ros::Duration(5.0).sleep();

      //Raise arms
      right_move_group.setNamedTarget("RIGHT_ARM_INITIAL_POSE");
      right_success = right_move_group.move();
    }
    else
    {
      left_move_group.setPoseTarget(grasp_pose);

      left_success = left_move_group.plan(left_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               left_success ? "SUCCEEDED" : "FAILED");

      // We can also visualize the plan as a line with markers in Rviz.
      ROS_INFO("Visualizing plan 1 as trajectory line");
      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "reach_pose");
      visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(left_arm_plan.trajectory_, left_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("Click Next");

      left_move_group.execute(left_arm_plan);

      //Reach movement
      left_move_group.setStartStateToCurrentState();
      grasp_pose.position.z = grasp_pose.position.z-0.07;
      left_move_group.setPoseTarget(grasp_pose);
      left_success = left_move_group.plan(left_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               left_success ? "SUCCEEDED" : "FAILED");
      // We can also visualize the plan as a line with markers in Rviz.
      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "pick_pose");
      visual_tools_ptr->publishText(text_pose, "Pick Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(left_arm_plan.trajectory_, left_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("next step");

      left_move_group.execute(left_arm_plan);

      //Remove object from the scene
      object_ids.push_back(req.object_name.data);
      planning_scene_interface.removeCollisionObjects(object_ids);

      //Close Gripper
      OperateLeftGripper(true);
      ros::Duration(3.0).sleep();

      //Reach movement
      left_move_group.setStartStateToCurrentState();
      grasp_pose.position.z = grasp_pose.position.z+0.12;
      left_move_group.setPoseTarget(grasp_pose);
      left_success = left_move_group.plan(left_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               left_success ? "SUCCEEDED" : "FAILED");
      // We can also visualize the plan as a line with markers in Rviz.
      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(grasp_pose, "reach_pose");
      visual_tools_ptr->publishText(text_pose, "Reach Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(left_arm_plan.trajectory_, left_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("next step");

      left_move_group.execute(left_arm_plan);

      //drop the ball
      left_move_group.setStartStateToCurrentState();
      place_pose.position.z = place_pose.position.z+0.4;

      ROS_INFO_STREAM(place_pose);
      //visual_tools_ptr->prompt("Click Next");

      left_move_group.setPoseTarget(place_pose);
      left_success = left_move_group.plan(left_arm_plan);
      ROS_INFO("Visualizing plan to target: %s",
               left_success ? "SUCCEEDED" : "FAILED");

      visual_tools_ptr->deleteAllMarkers();
      visual_tools_ptr->publishAxisLabeled(place_pose, "drop_pose");
      visual_tools_ptr->publishText(text_pose, "Drop Pose", rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
      visual_tools_ptr->publishTrajectoryLine(left_arm_plan.trajectory_, left_joint_model_group);
      visual_tools_ptr->trigger();
      visual_tools_ptr->prompt("next step");

      left_move_group.execute(left_arm_plan);


      //Open Gripper
      OperateLeftGripper(false);
      ros::Duration(5.0).sleep();

      //Raise arms
      left_move_group.setNamedTarget("LEFT_ARM_INITIAL_POSE");
      left_success = left_move_group.move();
    }
    res.success = true;
  }
}

bool PR2PickPlace::IsPickPoseWithinLimits(geometry_msgs::Pose &pick_pose,
                                          geometry_msgs::Pose &act_obj_pose)
{
  //Obtain actual pose of the object
  float dist_sq=0;

  dist_sq = (act_obj_pose.position.x-pick_pose.position.x)*(act_obj_pose.position.x-pick_pose.position.x)
    +(act_obj_pose.position.y-pick_pose.position.y)*(act_obj_pose.position.y-pick_pose.position.y)
    +(act_obj_pose.position.z-pick_pose.position.z)*(act_obj_pose.position.z-pick_pose.position.z);

  if(dist_sq < 0.03)
    return true;
  else
    return false;
}

bool PR2PickPlace::SetupCollisionObject(const std::string &object_id,
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

bool PR2PickPlace::OperateRightGripper(const bool &close_gripper)
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

bool PR2PickPlace::OperateLeftGripper(const bool &close_gripper)
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
    gripper_joint_positions[0] = 0.045;
    gripper_joint_positions[1] = 0.045;
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

tf::Quaternion PR2PickPlace::RPYToQuaternion(float R, float P, float Y)
{
  tf::Matrix3x3 mat;
  mat.setEulerYPR(Y,P,R);

  tf::Quaternion quat;
  mat.getRotation(quat);

  return quat;
}

PR2PickPlace::~PR2PickPlace(){}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "pr2_pick_place_server");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(8);
  spinner.start();
  PR2PickPlace pr2_pick_place(nh);
  ros::ServiceServer service = nh.advertiseService("pick_place_routine", &PR2PickPlace::Routine, &pr2_pick_place);
  ros::waitForShutdown();
  return 0;
}
