#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Eigen>

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
