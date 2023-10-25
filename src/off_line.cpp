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
#include <pcl/filters/filter.h>
#include <pcl/point_representation.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
// #include <pcl/concatenate.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <seg_ros/KeypointsWithScores.h>

void get_z_interval(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Eigen::Vector2d& z_interval);
void get_mask(const Eigen::MatrixX2d& keypoints, const Eigen::MatrixX3d& keypoints_pos, const std::vector<std::vector<int>>& human_dict, cv::Mat& mask_image);


int main(int argc, char** argv){
    ros::init(argc, argv, "truncator");
    ros::NodeHandle nh;

    std::string color_path = "/home/joy/mm_ws/src/seg_ros/images/color/color_2.png";
    std::string depth_path = "/home/joy/mm_ws/src/seg_ros/images/depth/depth_2.png";
    std::string seg_path = "/home/joy/mm_ws/src/seg_ros/images/seg/seg_2.png";
    std::string pose_path = "/home/joy/mm_ws/src/seg_ros/images/pose/json/pose_2.json";

    ros::Publisher pub_raw = nh.advertise<sensor_msgs::PointCloud2>("/pc_human", 5);
    ros::Publisher pub_part = nh.advertise<sensor_msgs::PointCloud2>("/pc_part", 5);
    ros::Publisher pub_markers = nh.advertise<visualization_msgs::MarkerArray>("/cylinder_marker", 5);

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
        scores(i) = scores_vec[i];
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
            
            // // 测试部位分割结果
            // if(mask_image.at<uchar>(m,n)!=20){
            //     continue;
            // }
            
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

            // if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)){
            //     continue;
            // }

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

    // 得到包含关节索引的human_dict
    std::vector<std::vector<int>> human_dict;
    human_dict.push_back({0,1,2,3,4,5,6});
    human_dict.push_back({5,6,11,12});
    human_dict.push_back({11,13});
    human_dict.push_back({12,14});
    human_dict.push_back({13,15});
    human_dict.push_back({14,16});
    human_dict.push_back({5,7});
    human_dict.push_back({6,8});
    human_dict.push_back({7,9});
    human_dict.push_back({8,10});

    // 绘制掩模图像（可以在一开始就绘制并直接生成部位点云）
    cv::Mat mask_image(depth_image.rows, depth_image.cols, CV_8UC1, cv::Scalar(255));
    get_mask(keypoints, keypoints_pos, human_dict, mask_image);
    
    // 遍历人体点云，并放入部位点云
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> keypart_cloud;
    for(int i = 0; i < 10; ++i) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        keypart_cloud.push_back(cloud);
    }
    
    for(const auto& point : pc_human->points){
        int n = static_cast<int>(point.x/point.z*fx+cx);
        int m = static_cast<int>(point.y/point.z*fy+cy);
        if(mask_image.at<uchar>(m,n) != 255){
            keypart_cloud[mask_image.at<uchar>(m,n)]->points.push_back(point);
        }
    }

    for(int i=0; i<10; i++){
        ROS_INFO("The pointcloud size of Keypart %i is: %li",i,keypart_cloud[i]->points.size());
    }

    // 圆柱点云的人体部位配准，以左上臂{5,7}为例
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    double spacing = 0.02;

    uint8_t r = 94;
    uint8_t g = 100;
    uint8_t b = 222;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

    double height = 0.3;
    double radius = 0.05;

    for (double z = -height / 2.0; z <= height / 2.0; z += spacing) {
        // 若为完整圆柱，采用 0 - 2.0 * M_PI
        for (double theta = - M_PI/2.0; theta <= M_PI/2.0; theta += spacing / radius) {
            pcl::PointXYZRGB point;
            point.x = radius * cos(theta);
            point.y = radius * sin(theta);
            point.z = z;

            point.rgb = *reinterpret_cast<float*>(&rgb);

            cylinder->points.push_back(point);
        }
    }

    Eigen::Vector3d p1 = keypoints_pos.row(7);
    Eigen::Vector3d p2 = keypoints_pos.row(9);

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    Eigen::Vector3d translation = (p1+p2) / 2.0;
    Eigen::Matrix4d translation_matrix = Eigen::Matrix4d::Identity();
    translation_matrix.block<3, 1>(0, 3) = translation;

    Eigen::Vector3d z_axis = (p1-p2).normalized();
    Eigen::Vector3d y_axis = translation.cross(z_axis).normalized();
    Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
    
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;

    transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;
    transformation_matrix = translation_matrix * transformation_matrix;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cylinder, *cylinder_trans, transformation_matrix);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_align(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(500);
    icp.setMaxCorrespondenceDistance(0.02);
    icp.setTransformationEpsilon(1e-4);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setInputSource(keypart_cloud[8]);
    icp.setInputTarget(cylinder_trans);
    icp.align(*cylinder_align);
    

    // pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
    // ne.setInputCloud(cylinder_trans);
    // ne.setRadiusSearch(0.1);
    // pcl::PointCloud<pcl::PointNormal>::Ptr cylinder_normal(new pcl::PointCloud<pcl::PointNormal>);
    // ne.compute(*cylinder_normal);

    // pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne1;
    // ne1.setInputCloud(keypart_cloud[8]);
    // ne1.setRadiusSearch(0.1);
    // pcl::PointCloud<pcl::PointNormal>::Ptr keypart_normal(new pcl::PointCloud<pcl::PointNormal>);
    // ne1.compute(*keypart_normal);

    // ROS_INFO("The size of Keypart Normal Cloud is: %li", cylinder_normal->points.size());
    // ROS_INFO("The size of Cylinder Normal Cloud is: %li", keypart_normal->points.size());

    // pcl::PointCloud<pcl::PointNormal>::Ptr cylinder_1(new pcl::PointCloud<pcl::PointNormal>);
    // pcl::PointCloud<pcl::PointNormal>::Ptr keypart_1(new pcl::PointCloud<pcl::PointNormal>);
    // pcl::PointCloud<pcl::PointNormal>::Ptr cylinder_2(new pcl::PointCloud<pcl::PointNormal>);
    // pcl::PointCloud<pcl::PointNormal>::Ptr keypart_2(new pcl::PointCloud<pcl::PointNormal>);

    // std::vector<int> mapping1, mapping2;
    // pcl::removeNaNNormalsFromPointCloud(*cylinder_normal,*cylinder_1,mapping1);
    // pcl::removeNaNNormalsFromPointCloud(*keypart_normal,*keypart_1,mapping2);

    // std::vector<int> mapping3, mapping4;
    // pcl::removeNaNNormalsFromPointCloud(*cylinder_1,*cylinder_2,mapping3);
    // pcl::removeNaNNormalsFromPointCloud(*keypart_1,*keypart_2,mapping4);

    // for(pcl::PointCloud<pcl::PointNormal>::iterator it = cylinder_normal->points.begin(); it != cylinder_normal->points.end(); it++){
    //     if(!pcl::isFinite(*it)){
    //         it = cylinder_normal->points.erase(it);
    //     }
    // }

    // for(pcl::PointCloud<pcl::PointNormal>::iterator it = keypart_normal->points.begin(); it != keypart_normal->points.end(); it++){
    //     if(!pcl::isFinite(*it)){
    //         it = keypart_normal->points.erase(it);
    //     }
    // }

    // ROS_INFO("The size of Keypart Finite Cloud is: %li", cylinder_normal->points.size());
    // ROS_INFO("The size of Cylinder Finite Cloud is: %li", keypart_normal->points.size());

    // pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;

    // icp.setMaximumIterations(500);
    // icp.setMaxCorrespondenceDistance(0.02);
    // icp.setTransformationEpsilon(1e-4);
    // icp.setEuclideanFitnessEpsilon(1e-4);
    // icp.setInputSource(cylinder_normal);
    // icp.setInputTarget(keypart_normal);

    // pcl::PointCloud<pcl::PointNormal>::Ptr aligned_source(new pcl::PointCloud<pcl::PointNormal>);
    // icp.align(*aligned_source);

    if (icp.hasConverged())
    {
        std::cout << "ICP converged. Score: " << icp.getFitnessScore() << std::endl;
        // 获取ICP的变换矩阵
        Eigen::Matrix4d part_cyli = icp.getFinalTransformation().cast<double>();
        ROS_INFO("ICP Trans Result--(%lf, %lf, %lf)",part_cyli(0,3),part_cyli(1,3),part_cyli(2,3));
    }
    else
    {
        std::cout << "ICP did not converge." << std::endl;
    }


    sensor_msgs::PointCloud2 msg_raw;
    sensor_msgs::PointCloud2 msg_part;

    visualization_msgs::Marker cylinder_marker;
    cylinder_marker.header.frame_id = "map";
    cylinder_marker.header.stamp = ros::Time::now();
    cylinder_marker.ns = "cylinder";
    cylinder_marker.id = 0;
    cylinder_marker.type = visualization_msgs::Marker::CYLINDER;
    cylinder_marker.action = visualization_msgs::Marker::ADD;
    Eigen::Matrix4d part_cyli = icp.getFinalTransformation().cast<double>();
    Eigen::Matrix4d cyli_map = transformation_matrix;
    Eigen::Matrix4d part_map = part_cyli * cyli_map;
    // Eigen::Matrix4d part_map = cyli_map; // 没有问题
    cylinder_marker.pose.position.x = part_map(0,3);
    cylinder_marker.pose.position.y = part_map(1,3);
    cylinder_marker.pose.position.z = part_map(2,3);
    Eigen::Quaterniond quat(part_map.block<3,3>(0,0));
    cylinder_marker.pose.orientation.x = quat.x();
    cylinder_marker.pose.orientation.y = quat.y();
    cylinder_marker.pose.orientation.z = quat.z();
    cylinder_marker.pose.orientation.w = quat.w();
    cylinder_marker.scale.x = 0.05*2;
    cylinder_marker.scale.y = 0.05*2;
    cylinder_marker.scale.z = 0.3;
    cylinder_marker.color.r = 94.0/255.0;
    cylinder_marker.color.g = 100.0/255.0;
    cylinder_marker.color.b = 222.0/255.0;
    cylinder_marker.color.a = 0.5;
    cylinder_marker.lifetime = ros::Duration();

    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.push_back(cylinder_marker);

    ros::Rate loop_rate(10.0);

    while(ros::ok()){
        pcl::toROSMsg(*cylinder_trans, msg_raw);
        // pcl::toROSMsg(*cylinder_align, msg_raw);
        // pcl::toROSMsg(*aligned_source, msg_raw);
        msg_raw.header.frame_id = "map";
        msg_raw.header.stamp = ros::Time::now();
        pub_raw.publish(msg_raw);

        pcl::toROSMsg(*keypart_cloud[8], msg_part);
        // pcl::toROSMsg(*cylinder_trans, msg_part);
        // pcl::toROSMsg(*pc_human, msg_part);
        msg_part.header.frame_id = "map";
        msg_part.header.stamp = ros::Time::now();
        pub_part.publish(msg_part);

        pub_markers.publish(marker_array);

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

void get_mask(const Eigen::MatrixX2d& keypoints, const Eigen::MatrixX3d& keypoints_pos, const std::vector<std::vector<int>>& human_dict, cv::Mat& mask_image){
    std::vector<cv::Point> trapezoid_body; // 躯干
    trapezoid_body.push_back(cv::Point((keypoints(6,0)-std::abs(keypoints(6,0)-keypoints(8,0))/2.0),keypoints(6,1)));
    trapezoid_body.push_back(cv::Point((keypoints(5,0)+std::abs(keypoints(5,0)-keypoints(7,0))/2.0),keypoints(5,1)));
    trapezoid_body.push_back(cv::Point((keypoints(5,0)+std::abs(keypoints(5,0)-keypoints(7,0))/2.0),keypoints(11,1)));
    trapezoid_body.push_back(cv::Point((keypoints(6,0)-std::abs(keypoints(6,0)-keypoints(8,0))/2.0),keypoints(12,1)));

    std::vector<cv::Point> trapezoid_head; // 头部
    trapezoid_head.push_back(cv::Point(keypoints(6,0),0));
    trapezoid_head.push_back(cv::Point(keypoints(5,0),0));
    trapezoid_head.push_back(cv::Point(keypoints(5,0),keypoints(5,1)));
    trapezoid_head.push_back(cv::Point(keypoints(6,0),keypoints(6,1)));

    cv::fillConvexPoly(mask_image, trapezoid_body, cv::Scalar(0));
    cv::fillConvexPoly(mask_image, trapezoid_head, cv::Scalar(1));


    for(int i=2; i<human_dict.size(); i++){
        Eigen::Vector2d a(keypoints(human_dict[i][0], 0), keypoints(human_dict[i][0], 1));
        Eigen::Vector2d b(keypoints(human_dict[i][1], 0), keypoints(human_dict[i][1], 1));
        double a_z = keypoints_pos(human_dict[i][0], 2);
        double b_z = keypoints_pos(human_dict[i][1], 2);
        double f = 912;
        double r = 0.10;
        if(i>5) {r = 0.05;}
        double a_r = r*f / a_z;
        double b_r = r*f / b_z;
        Eigen::Vector2d n;
        n(0) = a(1)-b(1);
        n(1) = b(0)-a(0);
        n.normalize();
        
        std::vector<cv::Point> trapezoid_limb;
        trapezoid_limb.push_back(cv::Point((a(0)-a_r*n(0)),(a(1)-a_r*n(1))));
        trapezoid_limb.push_back(cv::Point((a(0)+a_r*n(0)),(a(1)+a_r*n(1))));
        trapezoid_limb.push_back(cv::Point((b(0)+b_r*n(0)),(b(1)+b_r*n(1))));
        trapezoid_limb.push_back(cv::Point((b(0)-b_r*n(0)),(b(1)-b_r*n(1))));
        // 手部和足部的延长之后再考虑

        cv::fillConvexPoly(mask_image, trapezoid_limb, cv::Scalar(i));
    }

    // cv::imshow("Image", mask_image);
    // cv::waitKey(0);
}
