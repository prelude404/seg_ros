#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float32.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <json/json.h>
#include <algorithm>

// Eigen 库
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

// TF 库
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
 
// PCL 库
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
// #include <pcl/concatenate.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <seg_ros/KeypointsWithScores.h>

void get_z_interval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2d& z_interval);
void get_mask(const Eigen::MatrixX2d keypoints, const Eigen::MatrixX3d& keypoints_pos, cv::Mat& mask_image);

int main(int argc, char** argv){
    ros::init(argc, argv, "truncator");
    ros::NodeHandle nh;

    std::string color_path = "/home/joy/mm_ws/src/seg_ros/images/color/color_2.png";
    std::string depth_path = "/home/joy/mm_ws/src/seg_ros/images/depth/depth_2.png";
    std::string seg_path = "/home/joy/mm_ws/src/seg_ros/images/seg/seg_2.png";
    std::string pose_path = "/home/joy/mm_ws/src/seg_ros/images/pose/json/pose_2.json";

    ros::Publisher pub_raw = nh.advertise<sensor_msgs::PointCloud2>("/pc_human", 5);

    // 图像读取
    cv::Mat color_image = cv::imread(color_path, cv::IMREAD_UNCHANGED);
    cv::Mat depth_image = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    cv::Mat seg_image = cv::imread(seg_path, cv::IMREAD_UNCHANGED);

    std::ifstream pose_file(pose_path);

    Json::CharReaderBuilder reader;
    Json::Value root;
    std::string errs;
    Json::parseFromStream(reader, pose_file, &root, &errs);

    std::vector<std::vector<int>> keypoints_vec;
    if (root.isMember("keypoints") && root["keypoints"].isArray()) {
        const Json::Value& keypointsJson = root["keypoints"];
        for (const Json::Value& keypointJson : keypointsJson) {
            if (keypointJson.isArray()) {
                std::vector<int> keypoint;
                for (const Json::Value& value : keypointJson) {
                    if (value.isDouble()) {
                        keypoint.push_back(static_cast<int>(value.asDouble()));
                    }
                }
                keypoints_vec.push_back(keypoint);
            }
        }

        // 输出 keypoints 列表
        for (const std::vector<int>& keypoint : keypoints_vec) {
            for (int value : keypoint) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    Eigen::MatrixX2d keypoints(keypoints_vec.size(),2); // 关键点像素位置 (17*2)

    for (int i = 0; i < keypoints_vec.size(); ++i) {
        keypoints(i, 0) = keypoints_vec[i][0];
        keypoints(i, 1) = keypoints_vec[i][1];
    }
    

    // 从 JSON 数据中提取 scores 列表
    std::vector<double> scores_vec;
    if (root.isMember("scores") && root["scores"].isArray()) {
        const Json::Value& scoresJson = root["scores"];
        for (const Json::Value& value : scoresJson) {
            if (value.isDouble()) {
                scores_vec.push_back(value.asDouble());
            }
        }

        // // 输出 scores 列表
        // for (double score : scores_vec) {
        //     std::cout << score << " ";
        // }
        // std::cout << std::endl;
    }

    Eigen::VectorXd scores(scores_vec.size()); // 关键点置信度 (17*1)

    for (int i = 0; i < scores_vec.size(); ++i) {
        scores[i] = scores_vec[i];
    }

    // 相机1彩色内参
    double fx = 912.1243896484375;
    double fy = 912.0189819335938;
    double cx = 636.1998291015625;
    double cy = 381.69732666015625;
    double camera_factor = 1000; 

    // // 将深度图像进行高斯滤波（没用）
    // cv::Mat depth_image_smoothed;
    // cv::GaussianBlur(depth_image, depth_image_smoothed, cv::Size(5, 5), 0, 0);

    // m：短边，720，n：长边，1280
    ROS_INFO("rows:%i, cols:%i", depth_image.rows,depth_image.cols);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_raw(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(int m=0; m<depth_image.rows; m++){
        for(int n=0; n<depth_image.cols; n++){
            
            // 提取mm seg的人目标点云
            // int blue = seg_image.ptr<uchar>(m)[n*3];
            // seg_image为黑白图像，不用乘以3
            int blue = seg_image.at<uchar>(m,n);
            if(blue==0){
                continue; // 黑色RGB(0，0，0)
            }
            
            // 测试部位分割结果
            if(mask_image.at<uchar>(m,n)!=20){
                continue;
            }
            
            pcl::PointXYZRGB p;
            p.z =  depth_image.at<ushort>(m,n) / camera_factor;
            if(p.z==0.00){
                continue;
            }
            p.x = (n - cx) * p.z / fx;
            p.y = (m - cy) * p.z / fy;

            p.b = color_image.ptr<uchar>(m)[n*3];
            p.g = color_image.ptr<uchar>(m)[n*3+1];
            p.r = color_image.ptr<uchar>(m)[n*3+2];

            // ROS_INFO("point(%f,%f,%f)",p.z,p.x,p.y);

            pc_raw->points.push_back(p);

        }
    }
    
    double grid_size = 0.02;
    // 体素滤波
    pcl::VoxelGrid<pcl::PointXYZRGB> vox;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vox_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    vox.setInputCloud(pc_raw);
    vox.setLeafSize(grid_size, grid_size, grid_size);
    vox.filter(*vox_pc);

    // 对于人体点云进行欧式聚类
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(vox_pc);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02); // 设置点云之间的距离阈值
    ec.setMinClusterSize(10); // 设置点云簇的最小大小
    ec.setMaxClusterSize(999999); // 设置点云簇的最大大小
    ec.setSearchMethod(tree);
    ec.setInputCloud(vox_pc);
    ec.extract(cluster_indices);

    // 寻找最大的点云簇
    size_t largest_cluster_index = 0;
    size_t largest_cluster_size = 0;

    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        if (cluster_indices[i].indices.size() > largest_cluster_size) {
            largest_cluster_size = cluster_indices[i].indices.size();
            largest_cluster_index = i;
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_human(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(vox_pc);
    extract.setIndices(boost::make_shared<const pcl::PointIndices>(cluster_indices[largest_cluster_index]));
    extract.filter(*pc_human);

    ROS_INFO("Size of pc_human is %li", pc_human->size());

    // 获取深度方向包络盒区间
    Eigen::Vector2d z_interval;
    get_z_interval(pc_human, z_interval);
    ROS_INFO("Z_Interval of pc_human is [%lf, %lf]", z_interval[0], z_interval[1]);

    // 获取关键点的空间位置
    Eigen::MatrixX3d keypoints_pos(17,3); // 关键点空间位置 (17*3)
    
    for(int i=0; i<17; i++){
        int m = keypoints(i,1);
        int n = keypoints(i,0);
        keypoints_pos(i,2) = depth_image.at<ushort>(m,n) / camera_factor;
        keypoints_pos(i,0) = (n - cx) * keypoints_pos(i,2) / fx;
        keypoints_pos(i,1) = (m - cy) * keypoints_pos(i,2) / fy;

        keypoints_pos(i,2) = std::max(z_interval(0), std::min(z_interval(1),keypoints_pos(i,2)));
    }
    // std::cout << "keypoints_pos:\n" << keypoints_pos << std::endl;

    // 绘制掩模图像（可以在一开始就绘制并直接生成部位点云）
    cv::Mat mask_image(depth_image.rows, depth_image.cols, CV_8UC1, cv::Scalar(255));
    get_mask(keypoints, keypoints_pos, mask_image);
    
    
    
    
    
    
    sensor_msgs::PointCloud2 msg_raw;

    ros::Rate loop_rate(10.0);

    while(ros::ok()){
        pcl::toROSMsg(*pc_human, msg_raw);
        msg_raw.header.frame_id = "map";
        msg_raw.header.stamp = ros::Time::now();
        pub_raw.publish(msg_raw);

        ros::spinOnce();

        loop_rate.sleep();
    }

    return 0;
}

void get_z_interval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2d& z_interval){
    
    std::sort(cloud->points.begin(), cloud->points.end(), 
        [](const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2) {
            return point1.z < point2.z;
        });
    
    double delta = 0.90;
    size_t index_low = static_cast<size_t>((1-delta)/2.0 * cloud->points.size());
    size_t index_high = static_cast<size_t>((1+delta)/2.0 * cloud->points.size());
    
    z_interval[0] = cloud->points[index_low].z;
    z_interval[1] = cloud->points[index_high].z;
}

void get_mask(const Eigen::MatrixX2d keypoints, const Eigen::MatrixX3d& keypoints_pos, cv::Mat& mask_image){
    std::vector<cv::Point> trapezoid_body;
    trapezoid_body.push_back(cv::Point(0,0));
    trapezoid_body.push_back(cv::Point(mask_image.cols-1,0));
    trapezoid_body.push_back(cv::Point(mask_image.cols-1,std::max(keypoints(11,1),keypoints(12,1))));
    trapezoid_body.push_back(cv::Point(0,std::max(keypoints(11,1),keypoints(12,1))));

    std::vector<cv::Point> trapezoid_head;
    trapezoid_head.push_back(cv::Point(keypoints(6,0),0));
    trapezoid_head.push_back(cv::Point(keypoints(5,0),0));
    trapezoid_head.push_back(cv::Point(keypoints(5,0),keypoints(5,1)));
    trapezoid_head.push_back(cv::Point(keypoints(6,0),keypoints(6,1)));

    cv::fillConvexPoly(mask_image, trapezoid_body, cv::Scalar(0));
    cv::fillConvexPoly(mask_image, trapezoid_head, cv::Scalar(20));

    // cv::imshow("Image", mask_image);
    // cv::waitKey(0);
}