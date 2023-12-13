// 基础库
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
#include <algorithm>
#include <std_msgs/Float64MultiArray.h>

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
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <seg_ros/KeypointsWithScores.h>

int cam_num = 2;
int point_num = 17;
int part_num = 10;

bool depth_ready1 = false;
bool color_ready1 = false;
bool pose_ready1 = false;
bool depth_ready2 = false;
bool color_ready2 = false;
bool pose_ready2 = false;

class Camera
{
public:
    
    Camera(int value){
        cam_num = std::to_string(value);
        listener = nullptr;
        cam_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        keypoints.resize(point_num,2);
        scores.resize(point_num,1);
        positions.resize(3,point_num);
    }
    
    Camera(int value, tf::TransformListener *lis) : mask_image(480, 640, CV_8UC1, cv::Scalar(255)){
        cam_num = std::to_string(value);
        // tf::TransformListener *listener = lis; // 若在构造函数内定义listener则其作用域在该函数内
        listener = lis;
        cam_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        keypoints.resize(point_num,2);
        scores.resize(point_num,win_len);
        point_exist.resize(point_num,1);
        positions.resize(3,point_num);
    }

    std::string cam_num; // 相机编号
    double cx,cy,fx,fy; // 相机内参
    double camera_factor;
    cv::Mat color_pic; // 彩色图像
    cv::Mat depth_pic; // 深度图像
    cv::Mat mask_image; // 掩模图像
    cv::Mat color_mask; // 掩模附在彩色图像上
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam_pc,base_pc; // 相机和基坐标系点云
    Eigen::Matrix4d cam_to_base; // 储存cam2base
    void pic2cloud(); // 得到cam_pc, base_pc and cam_trans
    void pic2human(); // 得到人目标的实例分割点云
    void get_z_interval(); // 得到人点云的深度包围盒区间
    void calc_pos(); // 计算纠正后的关键点位置，并转至基坐标系
    void check_points(); // 判断每个关键点的可靠性

    Eigen::MatrixX2i keypoints;
    Eigen::MatrixXd scores;
    Eigen::VectorXi point_exist;
    Eigen::Matrix3Xd positions;
    Eigen::Vector2d z_interval;

    double z_far_lim = 2.0; // 原始点云的视野限制
    double grid_size = 0.02; // 体素滤波分辨率
    double z_delta = 0.90; // 深度包围盒点云比例
    int win_len = 5; // 滑窗的长度
    int r_len = 5; // 平均深度的方框
    double gamma = 0.7; // 遗忘因子
    double alpha = 2.0; // 限制力度

private:
    tf::TransformListener* listener; // 读取base2cam
    tf::StampedTransform cam_trans;
};

Camera cam1(1), cam2(2);

/***  彩色+深度图像转化为点云  ***/
void Camera::pic2cloud()
{   
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    // start = clock();
    // ROS_INFO("Cam_%s Depth_pic--Rows:%i,Cols:%i",cam_num.c_str(), depth_pic.rows, depth_pic.cols);
    // depth_pic.rows = 480; depth_pic.cols = 640;

    for (int m = 0; m < depth_pic.rows; m++)
    {
        for (int n = 0; n < depth_pic.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            float d = depth_pic.at<float>(m, n);//ushort d = depth_pic.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d <= 0.00)
                continue;
            // d 存在值，则向点云增加一个点
            pcl::PointXYZRGB p;

            // 考虑坐标系转换
            p.z = double(d) / camera_factor;
            p.x = (n - cx) * p.z / fx;
            p.y = (m - cy) * p.z / fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = color_pic.ptr<uchar>(m)[n*3];
            p.g = color_pic.ptr<uchar>(m)[n*3+1];
            p.r = color_pic.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            raw_pc->points.push_back(p);
        }
    }

    // if(listener != nullptr){ROS_INFO("Listener is NOT NULL");}
    try{
        listener->waitForTransform("base_link", ("cam_"+cam_num+"_link"),ros::Time(0.0),ros::Duration(1.0));
        listener->lookupTransform("base_link", ("cam_"+cam_num+"_link"),ros::Time(0.0),cam_trans);
    }
    catch(tf::TransformException &ex)
    {
        ROS_ERROR("camera_link: %s",ex.what());
        ros::Duration(0.5).sleep();
        return;
    }
    Eigen::Affine3d temp;
    tf::transformTFToEigen(cam_trans, temp);
    cam_to_base = temp.matrix().cast<double>();
    
    // std::cout << "CAM"+cam_num+" Trans: " << cam_to_base << std::endl;

    // cam_to_base = Eigen::Matrix4d::Identity();

    // 还需要运行时间和点云数量检测
    raw_pc->height = 1;
    raw_pc->width = raw_pc->points.size();
    ROS_INFO("[%s] Raw PointCloud Size = %i ",cam_num.c_str(),raw_pc->width);
    
    // 直通滤波
    pcl::PassThrough<pcl::PointXYZRGB> pass_z;
    pass_z.setInputCloud(raw_pc);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(0.0,z_far_lim);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_z_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pass_z.filter(*pass_z_pc);

    pass_z_pc->height = 1;
    pass_z_pc->width = pass_z_pc->points.size();
    ROS_INFO("[%s] Pass_Z PointCloud Size = %i ",cam_num.c_str(),pass_z_pc->width);

    // pcl::PassThrough<pcl::PointXYZRGB> pass_x;
    // pass_x.setInputCloud(pass_z_pc);
    // pass_x.setFilterFieldName("x");
    // pass_x.setFilterLimits(-1.3,1.3);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr pass_x_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pass_x.filter(*pass_x_pc);

    // pass_x_pc->height = 1;
    // pass_x_pc->width = pass_x_pc->points.size();
    // ROS_INFO("[%s] Pass_X PointCloud Size = %i ",cam_num.c_str(),pass_x_pc->width);
    
    // 体素滤波
    pcl::VoxelGrid<pcl::PointXYZRGB> vox;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vox_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    vox.setInputCloud(pass_z_pc);
    vox.setLeafSize(grid_size, grid_size, grid_size);
    vox.filter(*vox_pc);
    
    // ROS_INFO("PointCloud before Voxel Filtering: %i data points.",(raw_pc->width * raw_pc->height));
    vox_pc->height = 1;
    vox_pc->width = vox_pc->points.size();
    ROS_INFO("[%s] Voxel Filtered PointCloud Size = %i ",cam_num.c_str(),vox_pc->width);

    vox_pc->is_dense = false;

    // 【实机】转换到基坐标系下过滤桌面和机械臂！！！
    base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*vox_pc, *base_pc, cam_to_base);
    base_pc->height = 1;
    base_pc->width = base_pc->points.size();
    ROS_INFO("[%s] Base PointCloud Size = %i ",cam_num.c_str(),base_pc->width);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr desk_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass_desk;
    pass_desk.setInputCloud(base_pc);
    pass_desk.setFilterFieldName("z");
    pass_desk.setFilterLimits(0.05, 1.50);
    pass_desk.filter(*desk_pc);

    desk_pc->height = 1;
    desk_pc->width = desk_pc->points.size();
    ROS_INFO("[%s] Passed Table PointCloud Size = %i ",cam_num.c_str(),desk_pc->width);

    pcl::PassThrough<pcl::PointXYZRGB> pass_y;
    pass_y.setInputCloud(desk_pc);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(-1.0,1.0);
    base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pass_y.filter(*base_pc);

    base_pc->height = 1;
    base_pc->width = base_pc->points.size();
    ROS_INFO("[%s] Pass_Y PointCloud Size = %i ",cam_num.c_str(),base_pc->width);

    /* 圆柱包络去除机械臂 */

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam_y_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*base_pc, *cam_y_pc, cam_to_base.inverse().cast<float>());
    
    // 对于人体点云进行欧式聚类
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cam_y_pc);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.04); // 设置点云之间的距离阈值
    ec.setMinClusterSize(5); // 设置点云簇的最小大小
    ec.setMaxClusterSize(9999); // 设置点云簇的最大大小
    ec.setSearchMethod(tree);
    ec.setInputCloud(cam_y_pc);
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

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cam_y_pc);
    extract.setIndices(boost::make_shared<const pcl::PointIndices>(cluster_indices[largest_cluster_index]));
    cam_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    extract.filter(*cam_pc);

    cam_pc->height = 1;
    cam_pc->width = cam_pc->points.size();
    ROS_INFO("[%s] Camera PointCloud Size = %i ",cam_num.c_str(),cam_pc->width);


    // ROS_INFO("Size of cam keypoints is: %li x %li", keypoints.rows(), keypoints.cols()); // 17x2

    // base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::transformPointCloud(*cam_pc, *base_pc, cam_to_base); // cam_to_base

    // end = clock();
    // ROS_INFO("[%s] Total Running Time is: %f secs", cam_num.c_str(), static_cast<float>(end - start) / CLOCKS_PER_SEC);
}

/***  点云深度包围盒区间求取  ***/
void Camera::get_z_interval()
{
    std::sort(cam_pc->points.begin(), cam_pc->points.end(), 
        [](const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2) {
            return point1.z < point2.z;
        });

    size_t index_low = static_cast<size_t>((1-0.99)/2.0 * cam_pc->points.size());
    size_t index_high = static_cast<size_t>((1+0.50)/2.0 * cam_pc->points.size());
    
    z_interval[0] = cam_pc->points[index_low].z;
    z_interval[1] = cam_pc->points[index_high].z;

    // // 先令z_interval不发挥作用
    // z_interval[0] = cam_pc->points[0].z;
    // z_interval[1] = cam_pc->points[cam_pc->points.size()-1].z;

    ROS_INFO("Z_Interval of [cam%s] is [%lf, %lf]", cam_num.c_str(), z_interval[0], z_interval[1]);
}

/***  关键点的可靠性判定与深度估计与空间位置求取  ***/
void Camera::check_points()
{
    for(int i=0; i<point_num; i++){
        // 可靠性判定
        point_exist(i) = (scores.row(i).sum() > alpha) ? 1 : 0;

        // 深度估计：取相邻方块中的平均深度
        int m = keypoints(i,1);
        int n = keypoints(i,0);
        int start_row = std::max(m-r_len,0);
        int end_row = std::min(m+r_len,depth_pic.rows);
        int start_col = std::max(n-r_len,0);
        int end_col = std::min(n+r_len,depth_pic.cols);
        
        if(m<0 || m>=depth_pic.rows || n<0 || n>=depth_pic.cols){
            positions(2,i) = (z_interval(0)+z_interval(1))/2.0;
        }
        else{
            int count = 0;
            positions(2,i) = 0.0;
            for(int p=start_row; p<end_row; p++){
                for(int q=start_col; q<end_col; q++){
                    if(depth_pic.at<float>(p, q) <= 0.00) continue;
                    positions(2,i) += depth_pic.at<float>(p, q) / camera_factor;
                    // positions(2,i) += depth_pic.at<ushort>(p,q) / camera_factor;
                    count++;
                }
            }
            if(count>0){
                positions(2,i) /= count;
                positions(2,i) = std::max(z_interval(0), std::min(z_interval(1),positions(2,i)));
            }
            else{
                positions(2,i) = (z_interval(0)+z_interval(1))/2.0;
            }
        }

        // 获取深度图的数据类型: CV_32F
        // int depth_type = depth_pic.type();

        positions(0,i) = (n - cx) * positions(2,i) / fx;
        positions(1,i) = (m - cy) * positions(2,i) / fy;
    }
}

/***  关键点的空间位置求取和深度纠正  ***/
void Camera::calc_pos()
{
    for(int i=0; i<point_num; i++){

        point_exist(i) = (scores.row(i).sum() > alpha) ? 1 : 0;

        int m = keypoints(i,1);
        int n = keypoints(i,0);

        if(m<0 || m>depth_pic.rows || n<0 || n>depth_pic.cols){
            positions(2,i) = (z_interval(0)+z_interval(1))/2.0;
        }
        else{
            positions(2,i) = depth_pic.at<float>(m,n) / camera_factor;
            // positions(2,i) = depth_pic.ptr<float>(m)[n] / camera_factor;
            // positions(2,i) = depth_pic.at<ushort>(m,n) / camera_factor;
            // positions(2,i) = std::max(z_interval(0), std::min(z_interval(1),positions(2,i))); // 深度纠正（对于噪音非常有用！）
        }

        positions(0,i) = (n - cx) * positions(2,i) / fx;
        positions(1,i) = (m - cy) * positions(2,i) / fy;

        // ROS_INFO("Position of keypoint %i is: (%lf,%lf,%lf)", i, positions(0,i), positions(1,i), positions(2,i));
    }
}

class Window
{
public:
    Window(){}

    void add_state(const bool& exist, const Eigen::Matrix4d& trans_mat);
    
    void fuse_state(Eigen::Matrix4d& fused_trans);

private:
    int size_ = 5;
    std::deque<bool> exist_win;
    std::deque<Eigen::VectorXd> pose_win;

};

void Window::add_state(const bool& exist, const Eigen::Matrix4d& trans_mat){
    if(exist_win.size() == size_ && pose_win.size() == size_){
        exist_win.pop_front();
        pose_win.pop_front();
    }
    
    exist_win.push_back(exist);
    if(!exist){
        Eigen::VectorXd fake_state = Eigen::VectorXd::Zero(6);
        pose_win.push_back(fake_state);
    }
    else{
        Eigen::Matrix3d rotation_matrix = trans_mat.block<3, 3>(0, 0);
        Eigen::Vector3d translation_matrix = trans_mat.block<3, 1>(0, 3);
        Eigen::AngleAxisd rotation_angle_axis(rotation_matrix);
        Eigen::VectorXd real_state(6);
        real_state.head(3) = rotation_angle_axis.angle() * rotation_angle_axis.axis();
        real_state.tail(3) = translation_matrix;

        pose_win.push_back(real_state);
    }
}

void Window::fuse_state(Eigen::Matrix4d& fused_trans){
    
}

class BodyPart
{
public:
    BodyPart(std::string part_name, int part_type, std::vector<int> part_joints, double part_radius, double part_height) :
    point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>), cylinder_model(new pcl::PointCloud<pcl::PointXYZRGB>){
        name = part_name;
        joints = part_joints;
        exist = false;
        type = part_type;
        radius = part_radius;
        height = part_height;
    }

    std::string name;
    std::vector<int> joints;
    bool exist;

    double radius, height;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_model;

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cylinder;

    int type; // 1:head, 2:body, 3:legs and arms
};

class Human
{
public:

    Human(int value){
        human_num = std::to_string(value);
        keypoints_pos.resize(3,point_num);
        keypoints_score.resize(point_num,1);
        part_trans.resize(part_num);
        confidence = 0.3 * cam_num;
        human_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    }

    std::string human_num;
    Eigen::Matrix3Xd keypoints_pos;
    Eigen::VectorXd keypoints_score;

    std::vector<BodyPart> human_dict;
    void add_part(BodyPart& part){
        human_dict.push_back(part);
    }

    std::vector<Eigen::Matrix4d> part_trans;

    void fuse_pos(std::vector<Camera*> cams); // 得到各关键点的空间位置和置信度

    void get_part_cylinder(); // 得到每个部位的圆柱点云模型

    void check_parts(); // 判断每个部位是否在图像中出现（赋值exist）

    void extract_parts(std::vector<Camera*> cams); // 从每个相机的深度图像中提取并合并成部位点云

    void icp_cylinder(); // 生成圆柱点云与分离部位点云进行配准

    void pub_cylinder(std_msgs::Float64MultiArray& msg); // 发出圆柱阵列的位置姿态消息

    void get_markers(visualization_msgs::MarkerArray& marker_array); // 得到圆柱消息
    void update_markers(visualization_msgs::MarkerArray& marker_array); // 更新圆柱消息

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr human_pc;

private:
    double confidence;
};

Human human1(1);

/***  多相机融合关键点状态估计  ***/
void Human::fuse_pos(std::vector<Camera*> cams)
{
    keypoints_pos.setZero();
    keypoints_score.setZero();

    for(int i=0; i<point_num; i++){
        for(Camera* curr : cams){
            // 未在融合时考虑同一key point在不同相机中的存在性问题
            Camera& cam = (*curr);
            Eigen::Vector4d homogeneous_coordinates;
            homogeneous_coordinates << cam.positions.col(i), 1;
            Eigen::Vector4d base_coordinates = cam.cam_to_base * homogeneous_coordinates;
            keypoints_pos.col(i) += cam.scores(i,(cam.win_len-1)) * base_coordinates.head<3>();
            
            keypoints_score(i) += cam.scores(i,(cam.win_len-1));
        }
        keypoints_pos.col(i) /= keypoints_score(i);
        
        ROS_INFO("Position of keypoint %i is: (%lf,%lf,%lf)", i, keypoints_pos(0,i), keypoints_pos(1,i), keypoints_pos(2,i));
    }
}

/***  各部位用于配准的半圆柱点云  ***/
void Human::get_part_cylinder()
{
    double spacing = 0.02;

    // 为点设置颜色（这里用红色示例）
    uint8_t r = 255;
    uint8_t g = 0;
    uint8_t b = 0;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

    for(BodyPart& part : human_dict){

        int point_index = 0;

        double scale;

        switch (part.type){
            case 1:
                scale = 0.4;
                break;
            case 2:
                scale = 0.8;
                break;
            default:
                scale = 1.0;
                break;
        }
        
        // 计算圆柱上每个点的坐标和颜色并加入点云
        for (double z = -part.height / 2.0; z <= part.height / 2.0; z += spacing) {
            // 若为完整圆柱，采用 0 - 2.0 * M_PI（多台相机情况）
            for (double theta = - M_PI/2; theta <= M_PI/2; theta += spacing / part.radius) {
                pcl::PointXYZRGB point;
                point.x = scale * part.radius * cos(theta);
                point.y = part.radius * sin(theta);
                point.z = z;

                point.rgb = *reinterpret_cast<float*>(&rgb);

                part.cylinder_model->points.push_back(point);
            }
        }

        part.cylinder_model->height = 1;
        part.cylinder_model->width = part.cylinder_model->points.size();
        // ROS_INFO("[%s] Part Cylinder_model PointCloud Size = %i ",part.name.c_str(), part.cylinder_model->width);
    }
}

/***  各部位的存在性判断（最简单）  ***/
void Human::check_parts()
{
    // 先遍历各个parts对于各个parts分别判断是否存在
    for(BodyPart& part : human_dict){
        part.exist = false;
        for(const int& index : part.joints){
            if(keypoints_score[index] > confidence){
                part.exist = true;
                // ROS_INFO("Part %s Exist", part.name.c_str());
                break;
            }
        }
    }
}

/***  各部位的点云提取  ***/
void Human::extract_parts(std::vector<Camera*> cams)
{
    // 清空人体的点云
    for(BodyPart& part : human_dict){
        part.point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        part.exist = false;
    }
    
    // 分别获取每一相机的部位点云
    for(Camera* curr : cams){
        Camera& cam = (*curr);
        cam.mask_image.setTo(cv::Scalar(255));

        // 更新相机的图像坐标系关键点位置
        for(int i=0; i<point_num; i++){
            Eigen::Vector4d homogeneous_coordinates;
            homogeneous_coordinates << keypoints_pos.col(i),1;
            Eigen::Vector4d cam_coordinates = cam.cam_to_base.inverse() * homogeneous_coordinates;
            cam.positions.col(i) = cam_coordinates.head<3>();
            cam.keypoints(i,0) = static_cast<int>(cam.positions(0,i) * cam.fx / cam.positions(2,i) + cam.cx);
            cam.keypoints(i,1) = static_cast<int>(cam.positions(1,i) * cam.fy / cam.positions(2,i) + cam.cy);
        }

        // 制作掩模图像（存在性针对每台相机分别判断）
        for(int i=0; i<human_dict.size(); i++){
            
            BodyPart& part = human_dict[i];

            bool part_exist = false;

            // 判断该台相机中该部位的存在性
            for(const int& index : part.joints){
                if(cam.point_exist(index)==1){
                    part_exist = true;
                    part.exist = true;
                    break;
                }
            }

            if(!part_exist) continue;
            
            switch (part.type){
                case 1:{
                    std::vector<cv::Point> trapezoid_body; // 躯干
                    trapezoid_body.push_back(cv::Point((cam.keypoints(6,0)-std::abs(cam.keypoints(6,0)-cam.keypoints(8,0))/2.0),cam.keypoints(6,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));
                    trapezoid_body.push_back(cv::Point((cam.keypoints(5,0)+std::abs(cam.keypoints(7,0)-cam.keypoints(5,0))/2.0),cam.keypoints(5,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));
                    trapezoid_body.push_back(cv::Point((cam.keypoints(5,0)+std::abs(cam.keypoints(7,0)-cam.keypoints(5,0))/2.0),cam.keypoints(11,1)+std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));
                    trapezoid_body.push_back(cv::Point((cam.keypoints(6,0)-std::abs(cam.keypoints(6,0)-cam.keypoints(8,0))/2.0),cam.keypoints(12,1)+std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));
                    
                    cv::fillConvexPoly(cam.mask_image, trapezoid_body, cv::Scalar(0));
                    
                    break;
                }

                case 2:{
                    std::vector<cv::Point> trapezoid_head; // 头部
                    trapezoid_head.push_back(cv::Point(cam.keypoints(6,0)+std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/6.0,cam.keypoints(6,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))*1.5));
                    trapezoid_head.push_back(cv::Point(cam.keypoints(5,0)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/6.0,cam.keypoints(5,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))*1.5));
                    trapezoid_head.push_back(cv::Point(cam.keypoints(5,0)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/6.0,cam.keypoints(5,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));
                    trapezoid_head.push_back(cv::Point(cam.keypoints(6,0)+std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/6.0,cam.keypoints(6,1)-std::abs(cam.keypoints(5,0)-cam.keypoints(6,0))/4.0));

                    cv::fillConvexPoly(cam.mask_image, trapezoid_head, cv::Scalar(15));

                    break;
                }

                case 3:
                case 4:{
                    // Eigen::Vector2d a(cam.keypoints.row(part.joints[0]));
                    Eigen::Vector2d a,b;
                    a(0) = cam.keypoints(part.joints[0],0);
                    a(1) = cam.keypoints(part.joints[0],1);
                    // Eigen::Vector2d b(cam.keypoints.row(part.joints[1]));
                    b(0) = cam.keypoints(part.joints[1],0);
                    b(1) = cam.keypoints(part.joints[1],1);
                    double a_z = cam.positions(2,part.joints[0]);
                    double b_z = cam.positions(2,part.joints[1]);
                    double a_r = part.radius * cam.fx / a_z;
                    double b_r = part.radius * cam.fy / b_z;

                    Eigen::Vector2d n;
                    n(0) = a(1)-b(1);
                    n(1) = b(0)-a(0);
                    n.normalize();

                    if(part.type == 4){
                        b = b + (b-a)*0.7; // 手部和足部的延长
                    }

                    std::vector<cv::Point> trapezoid_limb;
                    trapezoid_limb.push_back(cv::Point((a(0)-a_r*n(0)),(a(1)-a_r*n(1))));
                    trapezoid_limb.push_back(cv::Point((a(0)+a_r*n(0)),(a(1)+a_r*n(1))));
                    trapezoid_limb.push_back(cv::Point((b(0)+b_r*n(0)),(b(1)+b_r*n(1))));
                    trapezoid_limb.push_back(cv::Point((b(0)-b_r*n(0)),(b(1)-b_r*n(1))));

                    cv::fillConvexPoly(cam.mask_image, trapezoid_limb, cv::Scalar(i*15));

                    break;
                }
            }
        }

        // 将mask_img附在color_pic上（通道数不同，分别融合再合并）
        std::vector<cv::Mat> channels_color;
        cv::split(cam.color_pic, channels_color);
        for (int i = 0; i < cam.color_pic.channels(); ++i) {
            cv::addWeighted(channels_color[i], 0.7, cam.mask_image, 0.3, 0, channels_color[i]);
        }
        cv::merge(channels_color, cam.color_mask);

        // 提取每一相机的每一部位的点云
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> keypart_cloud;
        for(int i = 0; i < part_num; ++i) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            keypart_cloud.push_back(cloud);
        }

        for(const auto& point : cam.cam_pc->points){
            int n = static_cast<int>(point.x/point.z*cam.fx+cam.cx);
            int m = static_cast<int>(point.y/point.z*cam.fy+cam.cy);

            if(m<0 || m>cam.mask_image.rows || n<0 || n>cam.mask_image.cols) continue;

            if(cam.mask_image.at<uchar>(m,n) != 255){
                keypart_cloud[cam.mask_image.at<uchar>(m,n)/15]->points.push_back(point);
            }
        }
        // 将点云转至base_link合并到human_dict
        for(int i=0; i<part_num; i++){
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*keypart_cloud[i], *trans_pc, cam.cam_to_base);
            *human_dict[i].point_cloud += *trans_pc;
        }
    }

    for(BodyPart& part : human_dict){
        part.point_cloud->height = 1;
        part.point_cloud->width = part.point_cloud->points.size();
        ROS_INFO("Part %s PointCloud Size = %i ",part.name.c_str(), part.point_cloud->width);
    }

    // 多个相机后可以对每个part进行欧式聚类！！
}

/***  各部位点云与半圆柱配准  ***/
void Human::icp_cylinder()
{
    human_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(500);
    icp.setMaxCorrespondenceDistance(0.02);
    icp.setTransformationEpsilon(1e-4);
    icp.setEuclideanFitnessEpsilon(1e-4);
    
    for(int i=0; i<part_num; i++){
        BodyPart& part = human_dict[i];

        if(part.point_cloud->points.size()<50){
            part.exist = false;
        }

        if(!part.exist) continue;

        // 求取初始的转换矩阵存入part_trans[i]
        switch(part.type){
            case 1:
            case 2:
            {
                // 使用点云坐标均值作为躯干中心
                Eigen::Vector4d centroid; // 齐次坐标
                pcl::compute3DCentroid(*part.point_cloud, centroid);
                Eigen::Vector3d translation = centroid.head<3>();
                
                // // 使用关键点平均位置作为躯干中心
                // Eigen::Vector3d translation = Eigen::Vector3d::Zero();
                // for(const int& index : part.joints){
                //     translation += keypoints_pos.col(index);
                // }
                // translation /= part.joints.size();

                part_trans[i] = Eigen::Matrix4d::Identity();
                part_trans[i].block<3,1>(0,3) = translation;
                Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
                Eigen::Vector3d y_axis = translation.cross(z_axis).normalized();
                Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();

                Eigen::Matrix3d rotation_matrix;
                rotation_matrix.col(0) = x_axis;
                rotation_matrix.col(1) = y_axis;
                rotation_matrix.col(2) = z_axis;
                Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
                transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;
                
                part_trans[i] = part_trans[i] * transformation_matrix;

                break;
            }
            case 3:
            case 4:
            {
                Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

                Eigen::Vector3d p1 = keypoints_pos.col(part.joints[0]);
                Eigen::Vector3d p2 = keypoints_pos.col(part.joints[1]);
                Eigen::Vector3d translation = (p1+p2) / 2.0;
                part_trans[i] = Eigen::Matrix4d::Identity();
                part_trans[i].block<3, 1>(0, 3) = translation;

                Eigen::Vector3d z_axis = (p1-p2).normalized();
                Eigen::Vector3d y_axis = translation.cross(z_axis).normalized();
                Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();
                
                Eigen::Matrix3d rotation_matrix;
                rotation_matrix.col(0) = x_axis;
                rotation_matrix.col(1) = y_axis;
                rotation_matrix.col(2) = z_axis;
                transformation_matrix.block<3, 3>(0, 0) = rotation_matrix;

                part_trans[i] = part_trans[i] * transformation_matrix;

                break;
            }
            default:
                break;
        }

        // if(part.point_cloud->points.size()<50) continue; // 报错：Not enough correspondences found. Relax your threshold parameters.

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*part.cylinder_model, *cylinder_trans, part_trans[i]);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_align(new pcl::PointCloud<pcl::PointXYZRGB>);

        icp.setInputSource(cylinder_trans);
        icp.setInputTarget(part.point_cloud);
        icp.align(*cylinder_align);

        if(i==0 || i==1){
            *human_pc += *cylinder_align; // 得到的人体模型点云（很不准确）
        }

        if (icp.hasConverged())
        {
            std::cout << "ICP converged. Score: " << icp.getFitnessScore() << std::endl;
            // 获取ICP的变换矩阵
            Eigen::Matrix4d cyli_part = icp.getFinalTransformation().cast<double>();
            ROS_INFO("Part [%s] ICP Trans Result--(%lf, %lf, %lf)",part.name.c_str(), cyli_part(0,3),cyli_part(1,3),cyli_part(2,3));

            part_trans[i] = cyli_part.inverse() * part_trans[i]; // 是否使用ICP配准的结果
        }
        else
        {
            std::cout << "ICP did not converge." << std::endl;
        }
    }
}

void Human::pub_cylinder(std_msgs::Float64MultiArray& msg)
{
    msg.data.clear();

    for(int i=0; i<part_num; i++){
        BodyPart& part = human_dict[i];
        if(!part.exist) continue;
        
        Eigen::Matrix4f part_base = (part_trans[i]).cast<float>();
        msg.data.push_back(part_base(0,3));
        msg.data.push_back(part_base(1,3));
        msg.data.push_back((part_base(2,3)-0.8));
        msg.data.push_back(part_base(0,2));
        msg.data.push_back(part_base(1,2));
        msg.data.push_back(part_base(2,2));
        msg.data.push_back(part.radius);
        msg.data.push_back(part.height);
        msg.data.push_back(0.0);
        msg.data.push_back(0.0);
        msg.data.push_back(0.0);
        msg.data.push_back(0.0);
        msg.data.push_back(0.0);
        msg.data.push_back(0.0);
    }
}

/***  最初获得要发布的圆柱消息  ***/
void Human::get_markers(visualization_msgs::MarkerArray& marker_array)
{
    marker_array.markers.clear();
    
    for(int i=0; i<part_num; i++){
        visualization_msgs::Marker cylinder_marker;
        cylinder_marker.header.frame_id = "base_link";
        cylinder_marker.header.stamp = ros::Time::now();
        cylinder_marker.ns = "cylinder";
        cylinder_marker.id = i;
        cylinder_marker.type = visualization_msgs::Marker::CYLINDER;
        cylinder_marker.action = visualization_msgs::Marker::ADD;

        // 初始位姿：随意定义
        cylinder_marker.pose.position.x = 0.0;
        cylinder_marker.pose.position.y = 0.0;
        cylinder_marker.pose.position.z = 0.0;
        cylinder_marker.pose.orientation.x = 0.0;
        cylinder_marker.pose.orientation.y = 0.0;
        cylinder_marker.pose.orientation.z = 0.0;
        cylinder_marker.pose.orientation.w = 1.0;

        // 几何尺寸：依据实际
        cylinder_marker.scale.x = 2.0 * human_dict[i].radius;
        cylinder_marker.scale.y = 2.0 * human_dict[i].radius;
        cylinder_marker.scale.z = human_dict[i].height;

        // 颜色：有颜色不透明
        cylinder_marker.color.r = 94.0/255.0;
        cylinder_marker.color.g = 100.0/255.0;
        cylinder_marker.color.b = 222.0/255.0;
        cylinder_marker.color.a = 0.5;
        
        cylinder_marker.lifetime = ros::Duration();

        marker_array.markers.push_back(cylinder_marker);
    }

}

/***  更改圆柱消息的位姿和存在性  ***/
void Human::update_markers(visualization_msgs::MarkerArray& marker_array)
{
    for(int i=0; i<part_num; i++){
        
        if(!human_dict[i].exist){
            marker_array.markers[i].color.a = 0.0; // 不存在部位仅更改透明度
        }
        else{
            // 存在部位需显示为不透明并实时更新位置和姿态
            marker_array.markers[i].color.a = 0.5;

            marker_array.markers[i].pose.position.x = part_trans[i](0,3);
            marker_array.markers[i].pose.position.y = part_trans[i](1,3);
            marker_array.markers[i].pose.position.z = part_trans[i](2,3);
            Eigen::Quaterniond quat(part_trans[i].block<3,3>(0,0));
            quat.normalize();

            marker_array.markers[i].pose.orientation.x = quat.x();
            marker_array.markers[i].pose.orientation.y = quat.y();
            marker_array.markers[i].pose.orientation.z = quat.z();
            marker_array.markers[i].pose.orientation.w = quat.w();
        }
    }
}

/***  CAM1 RGB处理  ***/
void color_cb1(const sensor_msgs::ImageConstPtr& color_msg)
{
    cv_bridge::CvImagePtr color_ptr;
    try
    {
        color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
        cv::waitKey(50); // 不断刷新图像，频率时间为int delay，单位为ms
    }
    catch (cv_bridge::Exception& e )
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
    }
    cam1.color_pic = color_ptr->image;
    // ROS_INFO("Color!!!");
    color_ready1 = true;
}

/***  CAM1 Depth处理  ***/
void depth_cb1(const sensor_msgs::ImageConstPtr& depth_msg)
{
    cv_bridge::CvImagePtr depth_ptr;
    try
    {
        depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        cv::waitKey(50);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
    }
 
    cam1.depth_pic = depth_ptr->image;
    // ROS_INFO("Depth!!!");
    depth_ready1 = true;
}

/***  CAM_1 Pose处理  ***/
void pose_cb1(const seg_ros::KeypointsWithScores& msg)
{
    cam1.scores.leftCols(cam1.win_len-1) = cam1.gamma * cam1.scores.rightCols(cam1.win_len-1);

    for(int i=0; i<point_num; i++){
        cam1.keypoints(i,0) = static_cast<int>(msg.keypoints[i].x);
        cam1.keypoints(i,1) = static_cast<int>(msg.keypoints[i].y);
        cam1.scores(i,(cam1.win_len-1)) = msg.keypoint_scores[i].data;
    }
    pose_ready1 = true;

    // std::cout << "cam1 scores: " << cam1.scores << std::endl;
}

/***  CAM2 RGB处理  ***/
void color_cb2(const sensor_msgs::ImageConstPtr& color_msg)
{
    cv_bridge::CvImagePtr color_ptr;
    try
    {
        color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
        cv::waitKey(50); // 不断刷新图像，频率时间为int delay，单位为ms
    }
    catch (cv_bridge::Exception& e )
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", color_msg->encoding.c_str());
    }
    cam2.color_pic = color_ptr->image;
    // ROS_INFO("Color!!!");
    color_ready2 = true;
}

/***  CAM2 Depth处理  ***/
void depth_cb2(const sensor_msgs::ImageConstPtr& depth_msg)
{
    cv_bridge::CvImagePtr depth_ptr;
    try
    {
        depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        cv::waitKey(50);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
    }
 
    cam2.depth_pic = depth_ptr->image;
    // ROS_INFO("Depth!!!");
    depth_ready2 = true;
}

/***  CAM_2 Pose处理  ***/
void pose_cb2(const seg_ros::KeypointsWithScores& msg)
{
    cam2.scores.leftCols(cam2.win_len-1) = cam2.gamma * cam2.scores.rightCols(cam2.win_len-1);

    for(int i=0; i<point_num; i++){
        cam2.keypoints(i,0) = static_cast<int>(msg.keypoints[i].x);
        cam2.keypoints(i,1) = static_cast<int>(msg.keypoints[i].y);
        cam2.scores(i,(cam1.win_len-1)) = msg.keypoint_scores[i].data;
    }
    pose_ready2 = true;

    // std::cout << "cam2 scores: " << cam2.scores << std::endl;
}


int main(int argc, char** argv){
    ros::init(argc, argv, "on_line");
    ros::NodeHandle nh;
    
    image_transport::ImageTransport it1(nh);
    image_transport::Subscriber sub1_color = it1.subscribe(("/cam_"+cam1.cam_num+"/color/image_raw"), 1, color_cb1);
    image_transport::Subscriber sub1_depth = it1.subscribe(("/cam_"+cam1.cam_num+"/aligned_depth_to_color/image_raw"), 1, depth_cb1);
    
    image_transport::ImageTransport it2(nh);
    image_transport::Subscriber sub2_color = it2.subscribe(("/cam_"+cam2.cam_num+"/color/image_raw"), 1, color_cb2);
    image_transport::Subscriber sub2_depth = it2.subscribe(("/cam_"+cam2.cam_num+"/aligned_depth_to_color/image_raw"), 1, depth_cb2);

    ros::Subscriber sub1_pose = nh.subscribe(("cam_"+cam1.cam_num+"/pose"),1,pose_cb1);
    ros::Subscriber sub2_pose = nh.subscribe(("cam_"+cam2.cam_num+"/pose"),1,pose_cb2);
    
    ros::Publisher pub_pc1 = nh.advertise<sensor_msgs::PointCloud2>("/pc_1", 1);
    ros::Publisher pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("/pc_2", 1);
    ros::Publisher pub_markers = nh.advertise<visualization_msgs::MarkerArray>("/cylinder_marker", 1);
    ros::Publisher pub_cylinders = nh.advertise<std_msgs::Float64MultiArray>("/obsState", 1);
    ros::Publisher pub_mask1 = nh.advertise<sensor_msgs::Image>("/mask_1", 1);
    ros::Publisher pub_mask2 = nh.advertise<sensor_msgs::Image>("/mask_2", 1);

    ros::Publisher pub_part = nh.advertise<sensor_msgs::PointCloud2>("/part_pc", 1);
    ros::Publisher pub_model = nh.advertise<sensor_msgs::PointCloud2>("/part_model", 1);

    tf::TransformListener* lis_cam1 = new(tf::TransformListener);
    tf::TransformListener* lis_cam2 = new(tf::TransformListener);
    
    cam1 = Camera(1,lis_cam1);
    cam2 = Camera(2,lis_cam2);

    // D435 CAM_1
    cam1.fx = 608.7494506835938;
    cam1.fy = 608.6277465820312;
    cam1.cx = 315.4583435058594;
    cam1.cy = 255.28733825683594;

    // D435 CAM_2
    cam2.fx = 606.3751831054688;
    cam2.fy = 604.959716796875;
    cam2.cx = 331.2972717285156;
    cam2.cy = 243.7368927001953;

    cam1.camera_factor = 1000;
    cam2.camera_factor = 1000;

    std::vector<Camera*> cams;
    cams.push_back(&cam1);
    cams.push_back(&cam2);

    // 人模型的维护
    BodyPart body("body", 1, {5,6,11,12}, 0.15, 0.4);
    BodyPart head("head", 2, {0,1,2,3,4,5,6}, 0.1, 0.2);

    BodyPart arm_left_upper("arm_left_upper", 3, {5,7}, 0.06, 0.3);
    BodyPart arm_left_lower("arm_left_lower", 4,{7,9}, 0.05, 0.3);
    BodyPart arm_right_upper("arm_right_upper", 3,{6,8}, 0.06, 0.3);
    BodyPart arm_right_lower("arm_right_lower", 4,{8,10}, 0.05, 0.3);
    
    BodyPart leg_left_upper("leg_left_upper", 3,{11,13}, 0.12, 0.3);
    BodyPart leg_left_lower("leg_left_lower", 4,{13,15}, 0.1, 0.3);
    BodyPart leg_right_upper("leg_right_upper", 3,{12,14}, 0.12, 0.3);
    BodyPart leg_right_lower("leg_right_lower", 4,{14,16}, 0.1, 0.3);

    human1.add_part(body);
    human1.add_part(head);
    human1.add_part(leg_left_upper);
    human1.add_part(leg_right_upper);
    human1.add_part(leg_left_lower);
    human1.add_part(leg_right_lower);
    human1.add_part(arm_left_upper);
    human1.add_part(arm_right_upper);
    human1.add_part(arm_left_lower);
    human1.add_part(arm_right_lower);

    human1.get_part_cylinder();

    visualization_msgs::MarkerArray marker_array;
    human1.get_markers(marker_array);

    sensor_msgs::PointCloud2 msg_pc1, msg_pc2, msg_part, msg_model;

    std_msgs::Float64MultiArray msg_cylinder;

    ros::Rate loop_rate(30.0);

    while(ros::ok() && (!depth_ready1 || !color_ready1 || !pose_ready1 || !depth_ready2 || !color_ready2 || !pose_ready2)){
        ros::spinOnce();
        ROS_INFO("Waiting for Image or Pose Message...");
        loop_rate.sleep();
    }

    while(ros::ok())
    {

        for(Camera* curr : cams){
            Camera& cam = (*curr);
            
            cam.pic2cloud(); // 生成人的点云

            cam.get_z_interval(); // 人的点云深度区间

            cam.check_points(); // 关键点的存在性与空间位置

            // cam.calc_pos(); // 计算关键点空间位置
        }

        pcl::toROSMsg(*cam1.cam_pc, msg_pc1);
        msg_pc1.header.frame_id = "cam_1_link";
        msg_pc1.header.stamp = ros::Time::now();
        pub_pc1.publish(msg_pc1); // 发布相机的整体人的点云

        pcl::toROSMsg(*cam2.cam_pc, msg_pc2);
        msg_pc2.header.frame_id = "cam_2_link";
        msg_pc2.header.stamp = ros::Time::now();
        pub_pc2.publish(msg_pc2); // 发布相机的整体人的点云

        human1.fuse_pos(cams); // 融合相机关键点信息

        // human1.check_parts(); // 判断各部位存在性

        human1.extract_parts(cams); // 提取存在部位点云

        // 发布掩模图像
        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv_image.image = cam1.color_mask;
        sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
        pub_mask1.publish(msg);
        cv_image.image = cam2.color_mask;
        msg = cv_image.toImageMsg();
        pub_mask2.publish(msg);

        human1.icp_cylinder(); // 配准部位点云与圆柱模型

        pcl::toROSMsg((*human1.human_dict[0].point_cloud + *human1.human_dict[1].point_cloud), msg_part);
        msg_part.header.frame_id = "base_link";
        msg_part.header.stamp = ros::Time::now();
        pub_part.publish(msg_part);

        pcl::toROSMsg(*human1.human_pc, msg_model);
        msg_model.header.frame_id = "base_link";
        msg_model.header.stamp = ros::Time::now();
        pub_model.publish(msg_model);

        human1.pub_cylinder(msg_cylinder); // 更新圆柱位姿消息

        pub_cylinders.publish(msg_cylinder);

        human1.update_markers(marker_array); // 更新圆柱marker信息

        pub_markers.publish(marker_array); // 发布人体圆柱消息
        
        ros::spinOnce();
        loop_rate.sleep();
    }
}