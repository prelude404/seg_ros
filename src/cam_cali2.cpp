/*
===================================================
Already Tested ! Calibration and TF Tree Simulation
===================================================
*/

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
cv_bridge::CvImagePtr color_ptr2;
bool color_ready2 = false;


// 相机内参矩阵
std::vector<cv::Mat> camera_matrix;

// ArUco字典和参数
int aruco_id = 0;
double aruco_size = 0.10; // m
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
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

/***  CAM2 RGB处理  ***/
void color_cb2(const sensor_msgs::ImageConstPtr& color_msg)
{
    try
    {
        color_ptr2 = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
        cv::waitKey(50); // 不断刷新图像，频率时间为int delay，单位为ms
    }
    catch (cv_bridge::Exception& e )
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
    }

    // ROS_INFO("Color!!!");
    color_ready2 = true;
}

bool detect_aruco(cv_bridge::CvImagePtr cv_ptr, Eigen::Matrix4d& mat, int i)
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

        cv::aruco::estimatePoseSingleMarkers(corners, aruco_size, camera_matrix[i], cv::Mat(), rvecs, tvecs);
        
        // ROS_INFO("Detected ID [%i]", ids[0]);

        cv::Vec3d rvec = rvecs[0];
        cv::Vec3d tvec = tvecs[0];

        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);
        Eigen::Matrix3d rotation_matrix;
        cv::cv2eigen(rot_mat, rotation_matrix);
        Eigen::Vector3d translation_vector(tvec[0], tvec[1], tvec[2]);
        
        // std::cout << "Translation Vector: " << translation_vector << std::endl;

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

void calc_mean(const std::vector<Eigen::Matrix4d>& vec, Eigen::Matrix4d& ans)
{
    std::vector<Eigen::AngleAxisd> rot;
    std::vector<Eigen::Vector3d> trans;

    for(int i=0; i<vec.size(); i++){
        Eigen::Matrix3d rotation_matrix = vec[i].block<3, 3>(0, 0);
        Eigen::Vector3d translation_matrix = vec[i].block<3, 1>(0, 3);
        Eigen::AngleAxisd rotation_angle_axis(rotation_matrix);
        rot.push_back(rotation_angle_axis);
        trans.push_back(translation_matrix);
    }

    // 计算旋转的均值
    Eigen::Quaterniond avg_rotation_quaternion(0, 0, 0, 0);
    for (const auto& angle_axis : rot) {
        Eigen::Quaterniond q(angle_axis);
        avg_rotation_quaternion = avg_rotation_quaternion.slerp(1.0 / rot.size(), q);
    }
    Eigen::AngleAxisd avg_rotation(avg_rotation_quaternion);

    // 计算平移的均值
    Eigen::Vector3d avg_translation(0, 0, 0);
    for (const auto& translation : trans) {
        avg_translation += translation / trans.size();
    }

    double error = 0.0;
    for (const Eigen::Vector3d& translation : trans) {
        error += pow((translation-avg_translation).norm(),2);
    }

    error = sqrt(error);
    std::cout << "Average ERROR is: " << error << std::endl;

    ans = Eigen::Matrix4d::Identity();
    ans.block<3, 3>(0, 0) = avg_rotation.toRotationMatrix();
    ans.block<3, 1>(0, 3) = avg_translation;

    std::cout << "Average Transformation Matrix [cam2_cam1]:" << ans << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cam_cali");
    ros::NodeHandle nh;

    // 打个草稿
    Eigen::Matrix4d cam1_world;
    Eigen::Matrix4d cam2_world;

    cam1_world << 
      0.5, -0.433013,     -0.75, -0.263071,
 0.866025,      0.25,  0.433013, -0.363586,
        0, -0.866025,       0.5,    -0.725,
        0,         0,         0,         1;

    cam2_world <<
     -0.5, -0.433013,     -0.75, -0.263071,
 0.866025,     -0.25, -0.433013,  0.303586,
        0, -0.866025,       0.5,    -0.725,
        0,         0,         0,         1;

    Eigen::Matrix4d real_cam2_cam1;
    real_cam2_cam1 = cam1_world.inverse() * cam2_world;

    std::cout << "Exp Real Result: " << real_cam2_cam1 << std::endl;

    /*** aruco码标定 ***/
    image_transport::ImageTransport it1(nh);
    image_transport::Subscriber sub_color1 = it1.subscribe(("/cam_1/color/image_raw"), 1, color_cb1);
    image_transport::ImageTransport it2(nh);
    image_transport::Subscriber sub_color2 = it2.subscribe(("/cam_2/color/image_raw"), 1, color_cb2);


    ros::Publisher pub_trans1 = nh.advertise<geometry_msgs::PoseStamped>("/cam_1/trans", 1);
    geometry_msgs::PoseStamped trans_msg1;
    ros::Publisher pub_trans2 = nh.advertise<geometry_msgs::PoseStamped>("/cam_2/trans", 1);
    geometry_msgs::PoseStamped trans_msg2;

    tf::TransformBroadcaster tf_broadcaster1, tf_broadcaster2;
    tf::StampedTransform trans1, trans2;

    Eigen::Matrix4d mat1 = Eigen::Matrix4d::Identity(); // cam1_to_aruco
    Eigen::Matrix4d mat2 = Eigen::Matrix4d::Identity(); // cam2_to_aruco

    // 两台相机的内参矩阵
    cv::Mat cam1_mat = (cv::Mat_<double>(3, 3) <<
                        608.7494506835938, 0.0, 315.4583435058594,
                        0.0, 608.6277465820312, 255.28733825683594,
                        0.0, 0.0, 1.0);

    cv::Mat cam2_mat = (cv::Mat_<double>(3, 3) <<
                        606.3751831054688, 0.0, 331.2972717285156,
                        0.0, 604.959716796875, 243.7368927001953,
                        0.0, 0.0, 1.0);

    // 工位的两台相机
    // cv::Mat cam1_mat = (cv::Mat_<double>(3, 3) <<
    //                     908.630615234375, 0.0, 636.9541625976562,
    //                     0.0, 908.7796020507812, 382.03594970703125,
    //                     0.0, 0.0, 1.0);

    // cv::Mat cam2_mat = (cv::Mat_<double>(3, 3) <<
    //                     912.1243896484375, 0.0, 636.1998291015625,
    //                     0.0, 912.0189819335938, 381.69732666015625,
    //                     0.0, 0.0, 1.0);

    camera_matrix.push_back(cam1_mat);
    camera_matrix.push_back(cam2_mat);

    int mean_time = 0;
    std::vector<Eigen::Matrix4d> result;
    Eigen::Matrix4d cam2_to_cam1;

    ros::Rate loop_rate(10.0);

    while(ros::ok() && (!color_ready1 || !color_ready2)){
        ros::spinOnce();
        ROS_INFO("Waiting for Color Image...");
        loop_rate.sleep();
    }

    ROS_INFO("Successfully get Color Image!");

    while(ros::ok()){
        
        if(detect_aruco(color_ptr1, mat1, 0) && detect_aruco(color_ptr2, mat2, 1)){
            
            // std::cout << "Trans Matrix[1]: " << mat1 << std::endl;
            // std::cout << "Trans Matrix[2]: " << mat2 << std::endl;
            
            get_trans(mat1, trans1, trans_msg1);
            get_trans(mat2, trans2, trans_msg2);

            // tf_broadcaster1.sendTransform(tf::StampedTransform(trans1,ros::Time::now(),"cam_1_link", "aruco_link"));
            
            pub_trans1.publish(trans_msg1);
            pub_trans2.publish(trans_msg2);
            
            // 读取10次两台相机之间的转换矩阵用于求均值
            if(mean_time < 500){
                Eigen::Matrix4d cam2_cam1 = mat1 * mat2.inverse();
                std::cout << "#" << mean_time << "# Trans Matrix[cam2_cam1]: " << cam2_cam1 << std::endl;
                result.push_back(cam2_cam1);
                mean_time++;
            }
            else{
                calc_mean(result, cam2_to_cam1);
                break;
            }

        }
        
        ros::spinOnce();
        
        loop_rate.sleep();
    }

    ROS_INFO("Successfully get cam2_cam1 transformation!");

    // /*** TF树发布 ***/
    // tf::TransformBroadcaster tf_broadcaster;
    // tf::StampedTransform cam2_to_cam1_trans;
    
    // Eigen::Vector3d pos = cam2_to_cam1.block<3,1>(0,3);
    // Eigen::Quaterniond quat = Eigen::Quaterniond(cam2_to_cam1.block<3,3>(0,0));
    // cam2_to_cam1_trans.setOrigin(tf::Vector3(pos(0),pos(1),pos(2)));
    // cam2_to_cam1_trans.setRotation(tf::Quaternion(quat.x(),quat.y(),quat.z(),quat.w()));

    // while (ros::ok())
    // {
    //     tf_broadcaster.sendTransform(tf::StampedTransform(cam2_to_cam1_trans,ros::Time::now(),"cam_1_link", "cam_2_link"));

    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }

    return 0;
}