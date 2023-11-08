#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sim_cam_frame");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10.0);
    
    tf::TransformBroadcaster tf_broadcaster;

    Eigen::Matrix4d cam1_to_world, cam2_to_world;
    tf::StampedTransform cam1_trans, cam2_trans;

    cam1_to_world<< 0.587395, -0.143131, -0.796543, -0.0447751,
                    0.8093, 0.103886, 0.578136, -0.402986,
                    0, -0.984236, 0.176858, -0.760546,
                    0, 0, 0, 1;
    
    cam2_to_world<< -0.587395, -0.143131, -0.796543, -0.0447751,
                    0.8093, -0.103886, -0.578136, 0.402986,
                    0, -0.984236, 0.176858, -0.760546,
                    0, 0, 0, 1;

    Eigen::Vector3d cam1_pos = cam1_to_world.block<3,1>(0,3);
    Eigen::Quaterniond cam1_quat = Eigen::Quaterniond(cam1_to_world.block<3,3>(0,0));
    cam1_trans.setOrigin(tf::Vector3(cam1_pos(0),cam1_pos(1),cam1_pos(2)));
    cam1_trans.setRotation(tf::Quaternion(cam1_quat.x(),cam1_quat.y(),cam1_quat.z(),cam1_quat.w()));

    Eigen::Vector3d cam2_pos = cam2_to_world.block<3,1>(0,3);
    Eigen::Quaterniond cam2_quat = Eigen::Quaterniond(cam2_to_world.block<3,3>(0,0));
    cam2_trans.setOrigin(tf::Vector3(cam2_pos(0),cam2_pos(1),cam2_pos(2)));
    cam2_trans.setRotation(tf::Quaternion(cam2_quat.x(),cam2_quat.y(),cam2_quat.z(),cam2_quat.w()));

    while (ros::ok())
    {
        // tf_broadcaster.sendTransform(tf::StampedTransform(cam1_trans,ros::Time::now(),"base_link", "cam_1_link"));
        // tf_broadcaster.sendTransform(tf::StampedTransform(cam2_trans,ros::Time::now(),"base_link", "cam_2_link"));
        tf_broadcaster.sendTransform(tf::StampedTransform(cam1_trans,ros::Time::now(),"world", "cam_1_link"));
        tf_broadcaster.sendTransform(tf::StampedTransform(cam2_trans,ros::Time::now(),"world", "cam_2_link"));

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
