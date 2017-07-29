# 3D Perception
Before starting any work on this project, please complete all steps for [Exercise 1, 2 and 3](https://github.com/udacity/RoboND-Perception-Exercises). At the end of Exercise-3 you have a pipeline that can identify points that belong to a specific object.

In this project, you must assimilate your work from previous exercises to successfully complete a tabletop pick and place operation using PR2.

The PR2 has been outfitted with an RGB-D sensor much like the one you used in previous exercises. This sensor however is a bit noisy, much like real sensors.

Given the cluttered tabletop scenario, you must implement a perception pipeline using your work from Exercises 1,2 and 3 to identify target objects from a so-called “Pick-List” in that particular order, pick up those objects and place them in corresponding dropboxes.

# Project Setup
For this setup, catkin_ws is the name of active ROS Workspace, if your workspace name is different, change the commands accordingly
If you do not have an active ROS workspace, you can create one by:

```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Now that you have a workspace, clone or download this repo into the src directory of your workspace:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Perception-Project.git
```
### Note: If you have the Kinematics Pick and Place project in the same ROS Workspace as this project, please remove the 'gazebo_grasp_plugin' directory from this project otherwise ignore this note. 

Now install missing dependencies using rosdep install:
```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```
Build the project:
```sh
$ cd ~/catkin_ws
$ catkin_make
```
Add following to your .bashrc file
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH
```

If you haven’t already, following line can be added to your .bashrc to auto-source all new terminals
```
source ~/catkin_ws/devel/setup.bash
```

To run the demo:
```sh
$ roslaunch pr2_robot pick_place_demo.launch
```
![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)



Once Gazebo is up and running, make sure you see following in the gazebo world:
- Robot

- Table arrangement

- Three target objects on the table

- Dropboxes on either sides of the robot


If any of these items are missing, please report as an issue on [the waffle board](https://waffle.io/udacity/robotics-nanodegree-issues).

In your RViz window, you should see the robot and a partial collision map displayed:

![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Proceed through the demo by pressing the ‘Next’ button on the RViz window when a prompt appears in your active terminal

The demo ends when the robot has successfully picked and placed all objects into respective dropboxes (though sometimes the robot gets excited and throws objects across the room!)

Close the terminal window using **ctrl+c** before restarting the demo.


# Steps to complete the project:
1. Launch the project demo to get an overview of the project itself.
2. Write a ros node and subscribe to /pr2/world/points topic. This topic contains noisy point cloud data that you must work with. 
3. Remove noise from the point cloud using Outlier Removal filter
4. Use voxelgrid downsampling and passthrough filter. Keep this result handy, as it will be used in later steps.
5. Use RANSAC plane fitting to remove points that belong to the table.
6. Apply Euclidean clustering to create separate clusters for individual items.
7. Now read the contents of the pick_list.yaml file. For each object perform object recognition. Obtain the centroid of the set of points that belong to that specific object.
8. Remember the point cloud from step-4? You now must subtract the recognized object points from that point cloud and publish it over /pr2/3d_map/points topic. This topic is read by Moveit!, which essentially treats the table and all objects on top of it except the recognized target as collidable map elements, allowing the robot to plan its trajectory.
9. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
10. Rotate the robot back to its original state.
11. Create a ROS Client for the “pick_place_routine” rosservice. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
12. If everything was done correctly, the selected arm will perform pick and place operation and display trajectory in the rviz window
13. Placing all the objects in their respective dropoff box signifies successful completion of this project. 

Once you have a working implementation you can launch the actual project by
```sh
$ roslaunch pr2_robot pick_place_project.launch
```

For all the step-by-step details on how to complete this project see the [RoboND 3D Perception Project Lesson]()
Note: The robot is a bit moody at times and might leave objects on the table or fling them across the room :D
As long as your pipeline performs succesful recognition, your project will be considered successful even if the robot feels otherwise!
