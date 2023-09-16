#include <ros/ros.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>

bool check_Cylinder(pcl::PointXYZRGB start_p, pcl::PointXYZRGB end_p, pcl::PointXYZRGB check_p, double radius);

int main(int argc, char** argv){
    ros::init(argc, argv, "truncator");
    ros::NodeHandle nh;

    ros::Publisher pub_human = nh.advertise<sensor_msgs::PointCloud2>("/pc_human", 1);
    ros::Publisher pub_68 = nh.advertise<sensor_msgs::PointCloud2>("/pc_68", 1);
    ros::Publisher pub_810 = nh.advertise<sensor_msgs::PointCloud2>("/pc_810", 1);
    
    // 图像读取
    cv::Mat color_image = cv::imread("/home/joy/mm_ws/src/seg_ros/images/color/color_13.png", cv::IMREAD_UNCHANGED);
    cv::Mat depth_image = cv::imread("/home/joy/mm_ws/src/seg_ros/images/depth/depth_13.png", cv::IMREAD_UNCHANGED);
    cv::Mat seg_image = cv::imread("/home/joy/mm_ws/src/seg_ros/images/seg/seg_13.png", cv::IMREAD_UNCHANGED);

    // C++读取json文件较复杂，先手动把joint坐标复制下来
    // 0索引：n；1索引：m
    Eigen::Vector2i joint6(static_cast<int>(424.16666666666674), static_cast<int>(94.16667222976685));
    Eigen::Vector2i joint8(static_cast<int>(390.83333333333326), static_cast<int>(210.83333444595337));
    Eigen::Vector2i joint10(static_cast<int>(465.83333333333326), static_cast<int>(194.16666841506958));

    ROS_INFO("joint6(%i,%i)",joint6[1],joint6[0]);
    ROS_INFO("joint6(%i,%i)",joint8[1],joint8[0]);
    ROS_INFO("joint6(%i,%i)",joint10[1],joint10[0]);

    // 人工设定的圆柱半径
    double r68 = 0.05;
    double r810 = 0.06;

    // 相机1彩色内参
    double fx = 608.7494506835938;
    double fy = 608.6277465820312;
    double cx = 315.4583435058594;
    double cy = 255.28733825683594;
    double camera_factor = 1000;

    // 三个关节的空间位置
    pcl::PointXYZRGB p6;
    p6.z = depth_image.at<ushort>(joint6[1],joint6[0]) / camera_factor;
    p6.x = (joint6[0] - cx) * p6.z / fx;
    p6.y = (joint6[1] - cy) * p6.z / fy;

    pcl::PointXYZRGB p8;
    p8.z = depth_image.at<ushort>(joint8[1],joint8[0]) / camera_factor;
    p8.x = (joint8[0] - cx) * p8.z / fx;
    p8.y = (joint8[1] - cy) * p8.z / fy;

    pcl::PointXYZRGB p10;
    p10.z = depth_image.at<ushort>(joint10[1],joint10[0]) / camera_factor;
    p10.x = (joint10[0] - cx) * p10.z / fx;
    p10.y = (joint10[1] - cy) * p10.z / fy;

    ROS_INFO("joint6(%f,%f,%f)",p6.z,p6.x,p6.y);
    ROS_INFO("joint8(%f,%f,%f)",p8.z,p8.x,p8.y);
    ROS_INFO("joint10(%f,%f,%f)",p10.z,p10.x,p10.y);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_human(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_68(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_810(new pcl::PointCloud<pcl::PointXYZRGB>);

    // m：短边，480，n：长边，640
    ROS_INFO("rows:%i, cols:%i", depth_image.rows,depth_image.cols);

    for(int m=0; m<depth_image.rows; m++){
        for(int n=0; n<depth_image.cols; n++){
            // 提取mm seg的人目标点云
            int blue = seg_image.ptr<uchar>(m)[n*3];
            // if(blue==0){
            //     continue; // 黑色RGB(0，0，0)
            // }
            pcl::PointXYZRGB p;
            p.z =  depth_image.at<ushort>(m,n) / camera_factor;
            p.x = (n - cx) * p.z / fx;
            p.y = (m - cy) * p.z / fy;

            p.b = color_image.ptr<uchar>(m)[n*3];
            p.g = color_image.ptr<uchar>(m)[n*3+1];
            p.r = color_image.ptr<uchar>(m)[n*3+2];

            // ROS_INFO("point(%f,%f,%f)",p.z,p.x,p.y);

            pc_human->points.push_back(p);

            // 提取mm pose的骨骼圆柱包络点云
            if(check_Cylinder(p6,p8,p,r68)){
                pc_68->points.push_back(p);
            }

            if(check_Cylinder(p8,p10,p,r810)){
                pc_810->points.push_back(p);
            }
        }
    }

    ROS_INFO("Size of pc_human is %li", pc_human->size());
    ROS_INFO("Size of pc_68 is %li", pc_68->size());
    ROS_INFO("Size of pc_810 is %li", pc_810->size());

    sensor_msgs::PointCloud2 msg_human, msg_68, msg_810;

    ros::Rate loop_rate(10.0);

    while(ros::ok()){
        pcl::toROSMsg(*pc_human, msg_human);
        msg_human.header.frame_id = "map";
        msg_human.header.stamp = ros::Time::now();
        pub_human.publish(msg_human);

        pcl::toROSMsg(*pc_68, msg_68);
        msg_68.header.frame_id = "map";
        msg_68.header.stamp = ros::Time::now();
        pub_68.publish(msg_68);

        pcl::toROSMsg(*pc_810, msg_810);
        msg_810.header.frame_id = "map";
        msg_810.header.stamp = ros::Time::now();
        pub_810.publish(msg_810);

        ros::spinOnce();

        loop_rate.sleep();
    }

    return 0;
}

bool check_Cylinder(pcl::PointXYZRGB a, pcl::PointXYZRGB b, pcl::PointXYZRGB s, double r){
    double ab = sqrt(pow((a.x - b.x), 2.0) + pow((a.y - b.y), 2.0) + pow((a.z - b.z), 2.0));
    double as = sqrt(pow((a.x - s.x), 2.0) + pow((a.y - s.y), 2.0) + pow((a.z - s.z), 2.0));
    double bs = sqrt(pow((s.x - b.x), 2.0) + pow((s.y - b.y), 2.0) + pow((s.z - b.z), 2.0));
    double cos_A = (pow(as, 2.0) + pow(ab, 2.0) - pow(bs, 2.0)) / (2 * ab*as);
    double sin_A = sqrt(1 - pow(cos_A, 2.0));
    double dis = as*sin_A;

    double inner_sab = (a.x-b.x)*(a.x-s.x) + (a.y-b.y)*(a.y-s.y) + (a.z-b.z)*(a.z-s.z);
    double inner_sba = (b.x-a.x)*(b.x-s.x) + (b.y-a.y)*(b.y-s.y) + (b.z-a.z)*(b.z-s.z);

    return dis<r && inner_sab>0 && inner_sba>0;
}