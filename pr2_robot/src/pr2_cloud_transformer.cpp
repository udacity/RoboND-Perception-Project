/*******************************************************************************
 * Copyright (C) 2017 Udacity Inc.
 *
 * This file is part of Robotic Arm: Pick and Place project for Udacity
 * Robotics nano-degree program
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Harsh Pandya

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

/*
* Brief:
* This node transforms point cloud from /camera_link frame to /world frame with noise
*/

class CloudTransformer
{
public:
  explicit CloudTransformer(ros::NodeHandle nh)
    : nh_(nh),
      demo_(0)
  {
    // Define Publishers and Subscribers here
    pcl_sub_ = nh_.subscribe("/camera/depth_registered/points", 1, &CloudTransformer::pclCallback, this);
    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/pr2/world/points", 1);

    buffer_.reset(new sensor_msgs::PointCloud2);
    buffer_->header.frame_id = "world";

    if(ros::param::get("/pr2_cloud_transformer/demo", demo_))
    {
      ROS_INFO_STREAM("Demo flag: "<<demo_);
    }
    else
    {
      ROS_ERROR("Failed to get demo flag");
    }
  }

private:
  ros::NodeHandle nh_;
  bool demo_;
  ros::Subscriber pcl_sub_;
  ros::Publisher pcl_pub_;
  tf::TransformListener listener_;
  sensor_msgs::PointCloud2::Ptr buffer_;

  void pclCallback(const sensor_msgs::PointCloud2ConstPtr& pcl_msg)
  {
    float standard_deviation = 0.025;
    listener_.waitForTransform("world", "camera_link", ros::Time::now(), ros::Duration(1.0));
    pcl_ros::transformPointCloud("world", *pcl_msg, *buffer_, listener_);

    if(demo_)
    {
      pcl_pub_.publish(buffer_);
    }

    else
    {
      // Add noise to buffer Point Cloud
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZRGB>);
      sensor_msgs::PointCloud2::Ptr noisy_buffer_ (new sensor_msgs::PointCloud2);

      pcl::fromROSMsg (*buffer_, *cloud);

      // Create the filtering object
      pcl::VoxelGrid<pcl::PointXYZRGB> sor;
      sor.setInputCloud (cloud);
      sor.setLeafSize (0.02, 0.02, 0.02);
      sor.filter (*cloud_out);

      cloud_filtered->points.resize (cloud_out->points.size ());
      cloud_filtered->header = cloud_out->header;
      cloud_filtered->width = cloud_out->width;
      cloud_filtered->height = cloud_out->height;


      boost::mt19937 rng;
      rng.seed (static_cast<unsigned int> (time (0)));
      boost::normal_distribution<> nd (0, standard_deviation);
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor (rng, nd);

      for (size_t i = 0; i < cloud_out->points.size (); ++i)
      {
        cloud_filtered->points[i].x = cloud_out->points[i].x + static_cast<float> (var_nor ());
        cloud_filtered->points[i].y = cloud_out->points[i].y + static_cast<float> (var_nor ());
        cloud_filtered->points[i].z = cloud_out->points[i].z + static_cast<float> (var_nor ());
        cloud_filtered->points[i].r = cloud_out->points[i].r;
        cloud_filtered->points[i].g = cloud_out->points[i].g;
        cloud_filtered->points[i].b = cloud_out->points[i].b;
      }


      *cloud_out = *cloud_filtered + *cloud;

      pcl::toROSMsg (*cloud_out, *noisy_buffer_);

      pcl_pub_.publish(noisy_buffer_);
    }
  }
};  // End of class CloudTransformer

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pr2_point_cloud_tf");
  ros::NodeHandle nh;

  CloudTransformer tranform_cloud(nh);

  // Spin until ROS is shutdown
  while (ros::ok())
    ros::spin();

  return 0;
}
