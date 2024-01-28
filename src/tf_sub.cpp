#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Eigen>
#include <tf/transform_datatypes.h>
#include <Eigen/Core>
#include <Eigen/Geometry>


void pose_cb1(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    // 处理接收到的相机位姿消息

    // 创建Transform对象
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z));
    tf::Quaternion quaternion(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
    transform.setRotation(quaternion);

    // 发布变换
    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link", "cam_1_link"));

    // // 直接使用标定的结果
    // Eigen::Matrix4d cam2_to_cam1;

    // cam2_to_cam1 <<
    //  0.490285, -0.356463, -0.795333,  0.495356,
    //  0.325133,  0.921468, -0.212567,  0.209024,
    //  0.808646, -0.154371,   0.56768,   0.44622,
    //         0,         0,         0,         1;
    
    // Eigen::Matrix4d cam1_to_cam2;
    // cam1_to_cam2 = cam2_to_cam1.inverse();

    // Eigen::Affine3d affineTransform(cam1_to_cam2);

    // Eigen::Quaterniond rotation(affineTransform.rotation());
    // Eigen::Vector3d translation(affineTransform.translation());

    // // Create a tf::Transform using the extracted rotation and translation
    // tf::Transform tfTransform;

    // // Set rotation in tf::Transform
    // tfTransform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));

    // // Set translation in tf::Transform
    // tfTransform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));

    // // 发布变换
    // static tf::TransformBroadcaster br;
    // br.sendTransform(tf::StampedTransform(tfTransform, ros::Time::now(), "cam_2_link", "cam_1_link"));

}

void pose_cb2(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    // 处理接收到的相机位姿消息

    // 创建Transform对象
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z));
    tf::Quaternion quaternion(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
    transform.setRotation(quaternion);

    // 发布变换
    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link", "cam_2_link"));

    ROS_INFO("Reiceved and Published TF information");
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "tf_sub");

    // 创建NodeHandle对象
    ros::NodeHandle nh;

    // 订阅/cam1_pose话题
    ros::Subscriber sub1 = nh.subscribe("/cam1_pose", 5, pose_cb1);

    ros::Subscriber sub2 = nh.subscribe("/cam2_pose", 5, pose_cb2);

    // 循环处理ROS回调函数
    ros::spin();

    return 0;
}
