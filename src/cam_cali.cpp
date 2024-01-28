// 基础库
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <vector>
#include <string>
#include <iostream>

// Eigen 库
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

// TF 库
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

// OpenCV库
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

cv_bridge::CvImagePtr color_ptr1;
bool color_ready1 = false;

bool esti_flag = false;

// 相机内参矩阵
cv::Mat camera_matrix;

// ArUco字典和参数
int aruco_id = 0;
double aruco_size = 0.05; // m
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
cv::Ptr<cv::aruco::DetectorParameters> parameter = cv::aruco::DetectorParameters::create();

/***  CAM1 RGB处理  ***/
void color_cb1(const sensor_msgs::ImageConstPtr& color_msg)
{
    try
    {
        color_ptr1 = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
        cv::waitKey(50); // 不断刷新图像，频率时间为int delay，单位为ms
    }
    catch (cv_bridge::Exception& e )
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
    }

    // ROS_INFO("Color!!!");
    color_ready1 = true;
}

/*** 开始估计信号处理 ***/
void flag_cb(const std_msgs::Bool::ConstPtr& flag_msg)
{
    esti_flag = flag_msg->data;
}

bool detect_aruco(cv_bridge::CvImagePtr cv_ptr, Eigen::Matrix4d& mat)
{
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    
    cv::aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, parameter);
    
    if(ids.empty() || ids[0]!=aruco_id){
        ROS_INFO("Camera NOT Detect ArUco!!!");
        return false;
    }
    else{
        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::estimatePoseSingleMarkers(corners, aruco_size, camera_matrix, cv::Mat(), rvecs, tvecs);
        
        ROS_INFO("Detected ID [%i]", ids[0]);

        cv::Vec3d rvec = rvecs[0];
        cv::Vec3d tvec = tvecs[0];

        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);
        Eigen::Matrix3d rotation_matrix;
        cv::cv2eigen(rot_mat, rotation_matrix);
        Eigen::Vector3d translation_vector(tvec[0], tvec[1], tvec[2]);
        
        std::cout << "Translation Vector: " << translation_vector << std::endl;

        mat.block<3, 3>(0, 0) = rotation_matrix;
        mat.block<3, 1>(0, 3) = translation_vector;
    }

    return true;
}

void get_trans(const Eigen::Matrix4d& mat, tf::StampedTransform& trans, geometry_msgs::PoseStamped& trans_msg)
{
    Eigen::Vector3d pos = mat.block<3,1>(0,3);
    Eigen::Quaterniond quat = Eigen::Quaterniond(mat.block<3,3>(0,0));
    quat.normalize();
    trans.setOrigin(tf::Vector3(pos(0),pos(1),pos(2)));
    trans.setRotation(tf::Quaternion(quat.x(),quat.y(),quat.z(),quat.w()));

    trans_msg.header.stamp = ros::Time::now();
    trans_msg.pose.position.x = mat(0,3);
    trans_msg.pose.position.y = mat(1,3);
    trans_msg.pose.position.z = mat(2,3);
    trans_msg.pose.orientation.x = quat.x();
    trans_msg.pose.orientation.y = quat.y();
    trans_msg.pose.orientation.z = quat.z();
    trans_msg.pose.orientation.w = quat.w();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cam_cali");
    ros::NodeHandle nh;

    image_transport::ImageTransport it1(nh);
    image_transport::Subscriber sub_color1 = it1.subscribe(("/cam_2/color/image_raw"), 1, color_cb1);

    ros::Subscriber sub_flag = nh.subscribe<std_msgs::Bool>("/estimate_flag", 1, flag_cb);

    ros::Publisher pub_trans1 = nh.advertise<geometry_msgs::PoseStamped>("/cam_1/trans", 1);
    geometry_msgs::PoseStamped trans_msg1;

    tf::TransformBroadcaster tf_broadcaster1;
    tf::StampedTransform trans1;

    Eigen::Matrix4d mat1 = Eigen::Matrix4d::Identity();

    // 两台相机的内参矩阵
    // camera_matrix = (cv::Mat_<double>(3, 3) <<
    //                     608.7494506835938, 0.0, 315.4583435058594,
    //                     0.0, 608.6277465820312, 255.28733825683594,
    //                     0.0, 0.0, 1.0);

    camera_matrix = (cv::Mat_<double>(3, 3) <<
                        606.3751831054688, 0.0, 331.2972717285156,
                        0.0, 604.959716796875, 243.7368927001953,
                        0.0, 0.0, 1.0);

    ros::Rate loop_rate(30.0), true_rate(1.0);

    while(ros::ok() && (!color_ready1)){
        ros::spinOnce();
        ROS_INFO("Waiting for Color Image...");
        loop_rate.sleep();
    }

    ROS_INFO("Successfully get Color Image!");

    while(ros::ok()){
        
        if(esti_flag && detect_aruco(color_ptr1, mat1)){
            
            std::cout << "Trans Matrix1: " << mat1 << std::endl;
            
            get_trans(mat1, trans1, trans_msg1);

            // tf_broadcaster1.sendTransform(tf::StampedTransform(trans1,ros::Time::now(),"cam_1_link", "aruco_link"));
            
            pub_trans1.publish(trans_msg1);
            
            esti_flag = false;

            true_rate.sleep();
        }
        
        ros::spinOnce();
        
        loop_rate.sleep();
    }
}