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

class Camera
{
public:
    
    Camera(int value){
        cam_num = std::to_string(value);
        listener = nullptr;
        cam_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    }
    
    Camera(int value, tf::TransformListener *lis){
        cam_num = std::to_string(value);
        // tf::TransformListener *listener = lis; // 若在构造函数内定义listener则其作用域在该函数内
        listener = lis;
        cam_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    }

    std::string cam_num; // 相机编号
    double cx,cy,fx,fy; // 相机内参
    double camera_factor = 1000;
    cv::Mat color_pic, depth_pic; // 彩色和深度图像
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam_pc,base_pc; // 相机和基坐标系点云
    Eigen::Matrix4d cam_to_base; // 储存cam2base
    void pic2cloud(); // 得到cam_pc, base_pc and cam_trans
    void pic2human(); // 得到人目标的实例分割点云
    double z_far_lim = 5.0;
    int pc_num_lim = 9999;
    double grid_size = 0.03;

    std::vector<std::vector<double>> keypoints;
    std::vector<double> keypoint_scores;

private:
    tf::TransformListener* listener; // 读取base2cam
};

Camera cam1(1);

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
            float d = depth_pic.ptr<float>(m)[n];//ushort d = depth_pic.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            pcl::PointXYZRGB p;

            // // 相机模型是垂直的
            // p.x = double(d) / camera_factor;
            // p.y = -(n - camera_cx) * p.x / camera_fx;
            // p.z = -(m - camera_cy) * p.x / camera_fy;

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
            raw_pc->points.push_back( p );
        }
    }

    // if(listener != nullptr){ROS_INFO("Listener is NOT NULL");}
    tf::StampedTransform cam_trans;
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
    
    // // 随机采样
    // // int result = (a > b) ? b : c;
    // bool random_flag = (cam_pc->width > pc_num_lim);
    // if(random_flag)
    // {
    //     pcl::RandomSample<pcl::PointXYZRGB> rs;
    //     rs.setInputCloud(cam_pc);
    //     rs.setSample(pc_num_lim);
    //     rs.filter(*cam_pc);
        
    //     cam_pc->height = 1;
    //     cam_pc->width = cam_pc->points.size();
    //     // ROS_INFO("Random Sampled PointCloud Size = %i ",cam_pc->width);
    // }
    
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

    // cam_pc.reset();
    cam_pc = vox_pc;

    base_pc.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cam_pc, *base_pc, cam_to_base); // cam_to_base

    // end = clock();
    // ROS_INFO("[%s] Total Running Time is: %f secs", cam_num.c_str(), static_cast<float>(end - start) / CLOCKS_PER_SEC);

}

class BodyPart
{
public:
    BodyPart(std::string part_name, int part_type, std::vector<int> part_joints, double part_radius) :
    point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>), cylinder_model(new pcl::PointCloud<pcl::PointXYZRGB>), trans_cylinder(new pcl::PointCloud<pcl::PointXYZRGB>){
        name = part_name;
        joints = part_joints;
        exist = false;
        type = part_type;
        radius = part_radius;
    }

    std::string name;
    std::vector<int> joints;
    bool exist;

    std::vector<double> start_p, end_p;
    double radius;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_model;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cylinder;

    // void icp_Cylinder(); // 生成圆柱点云与分离部位点云进行配准

    int type; // 1:head, 2:body, 3:legs and arms
};

class Human
{
public:
    Human(int value) :  keypoints(17,{-1,-1}), keypoint_scores(17, 0.0), keypoint_pos(17,{0.0,0.0,0.0}) {
        human_num = std::to_string(value);
        confidence = 0.5;
    }
    std::string human_num;
    
    std::vector<std::vector<int>> keypoints; // 关键点在彩色图像上的预测位置
    std::vector<double> keypoint_scores; // 关键点预测的可信度
    std::vector<std::vector<double>> keypoint_pos; // 关键点在相机坐标系下的位置

    std::vector<BodyPart> human_dict;

    visualization_msgs::MarkerArray cylinder_model;

    void add_part(BodyPart& part){
        human_dict.push_back(part);
    }

    void get_part_cylinder(); // 得到每个部位的圆柱点云模型

    void check_parts(); // 判断每个部位是否在图像中出现（赋值exist）

    void seg_cloud(const Camera& cam); // 切割出每个存在部位的点云（赋值point_cloud）

    void get_model(); // 得到最终的由圆柱组成的人模型

private:
    double confidence;

    void calc_Cylinder(BodyPart& part);
    bool check_Cylinder(std::vector<double> a, std::vector<double> b, std::vector<double> s, double r);

};

Human human1(1);

void Human::check_parts(){
    // 先遍历各个parts对于各个parts分别判断是否存在
    for(BodyPart& part : human_dict){
        part.exist = false;
        for(const int& index : part.joints){
            if(keypoint_scores[index] > confidence){
                part.exist = true;
                break;
            }
        }
    }
}

void Human::seg_cloud(const Camera& cam){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cam.cam_pc, *new_pc); // 将cam.cam_pc复制给new_pc，可以删除其中的点云
    
    check_parts(); // 检查各个部位是否存在

    for(BodyPart& part : human_dict){
        if(!part.exist || part.type!=1) continue;

        calc_Cylinder(part); // 更新存在部位圆柱几何描述的参数
        
        part.point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::PointCloud<pcl::PointXYZRGB>::iterator it = new_pc->begin();
        while (it != new_pc->end()) {
            pcl::PointXYZRGB p = *it; // 获取当前点
            std::vector<double> p_v= {p.x, p.y, p.z};
            if(check_Cylinder(part.start_p, part.end_p, p_v, 1.0 * part.radius)){
                part.point_cloud->points.push_back(p);
                it = new_pc->erase(it);
            }
            else{
                ++it;
            }
            
        }
        
        // pcl::PointCloud<pcl::PointXYZRGB>::iterator it;
        // for (it = new_pc->begin(); it != new_pc->end(); ++it) {
        //     pcl::PointXYZRGB p = *it; // 获取当前点
        //     std::vector<double> p_v= {p.x, p.y, p.z};
        //     if(check_Cylinder(part.start_p, part.end_p, p_v, 1.0 * part.radius)){
        //         part.point_cloud->points.push_back(p);
        //     }
            
        //     // 可以执行！说明是check_Cylinder有问题！！！
        //     // if(true){
        //     //     part.point_cloud->points.push_back(p);
        //     // }
        // }

        // // 全部为0！！！需要解决！！！
        // part.point_cloud->height = 1;
        // part.point_cloud->width = part.point_cloud->points.size();
        // ROS_INFO("[%s] PointCloud Size = %i ",part.name.c_str(),part.point_cloud->width);

    }


    for(BodyPart& part : human_dict){
        if(!part.exist || part.type!=3) continue;

        calc_Cylinder(part); // 更新存在部位圆柱几何描述的参数
        
        part.point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::PointCloud<pcl::PointXYZRGB>::iterator it = new_pc->begin();
        while (it != new_pc->end()) {
            pcl::PointXYZRGB p = *it; // 获取当前点
            std::vector<double> p_v= {p.x, p.y, p.z};
            if(check_Cylinder(part.start_p, part.end_p, p_v, 1.0 * part.radius)){
                part.point_cloud->points.push_back(p);
                it = new_pc->erase(it);
            }
            else{
                ++it;
            }
            
        }

    }

    for(BodyPart& part : human_dict){
        if(!part.exist || part.type!=2) continue;

        calc_Cylinder(part); // 更新存在部位圆柱几何描述的参数
        
        part.point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::PointCloud<pcl::PointXYZRGB>::iterator it = new_pc->begin();
        while (it != new_pc->end()) {
            pcl::PointXYZRGB p = *it; // 获取当前点
            std::vector<double> p_v= {p.x, p.y, p.z};
            if(check_Cylinder(part.start_p, part.end_p, p_v, 1.0 * part.radius)){
                part.point_cloud->points.push_back(p);
                it = new_pc->erase(it);
            }
            else{
                ++it;
            }
            
        }

        // 仅保留部位点云的最大聚类
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(part.point_cloud);
        // 设置分割参数, 执行欧式聚类分割
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(0.03);  // 设置近邻搜索的半径
        ec.setMinClusterSize(10);     // 设置最小聚类点数
        ec.setMinClusterSize(99999);     // 设置最大聚类点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(part.point_cloud);
        ec.extract(cluster_indices);
        
        int largest_cluster_index = -1;
        size_t largest_cluster_size = 0;

        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            if (cluster_indices[i].indices.size() > largest_cluster_size) {
                largest_cluster_size = cluster_indices[i].indices.size();
                largest_cluster_index = i;
            }
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr largest_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(part.point_cloud);
        extract.setIndices(boost::make_shared<std::vector<int>>(cluster_indices[largest_cluster_index].indices));
        extract.setNegative(false);
        extract.filter(*largest_cluster);

        part.point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        part.point_cloud = largest_cluster;

    }
}

void Human::calc_Cylinder(BodyPart& part){
    if(part.type==1){
        // vector只是一种数组容器，不具备任何数学运算的功能！早知道一开始就用Eigen了！！
        Eigen::Vector3d temp1((keypoint_pos[2][0] - keypoint_pos[1][0]), (keypoint_pos[2][1] - keypoint_pos[1][1]), (keypoint_pos[2][2] - keypoint_pos[1][2]));
        Eigen::Vector3d temp2((keypoint_pos[6][0] - keypoint_pos[5][0]), (keypoint_pos[6][1] - keypoint_pos[5][1]), (keypoint_pos[6][2] - keypoint_pos[5][2]));

        part.start_p = {(keypoint_pos[1][0] + keypoint_pos[2][0])/2.0, (keypoint_pos[1][1] + keypoint_pos[2][1])/2.0, (keypoint_pos[1][2] + keypoint_pos[2][2])/2.0};
        part.end_p = {(keypoint_pos[5][0] + keypoint_pos[6][0])/2.0, (keypoint_pos[5][1] + keypoint_pos[6][1])/2.0, (keypoint_pos[5][2] + keypoint_pos[6][2])/2.0};
        part.start_p = {part.start_p[0]*2-part.end_p[0],part.start_p[1]*2-part.end_p[1],part.start_p[2]*2-part.end_p[2]};
        // part.radius = (temp1.norm() + temp2.norm()) / 4.0;
    }
    else if(part.type==2){
        part.start_p = {(keypoint_pos[5][0] + keypoint_pos[6][0])/2.0, (keypoint_pos[5][1] + keypoint_pos[6][1])/2.0, (keypoint_pos[5][2] + keypoint_pos[6][2])/2.0};
        part.end_p = {(keypoint_pos[11][0] + keypoint_pos[12][0])/2.0, (keypoint_pos[11][1] + keypoint_pos[12][1])/2.0, (keypoint_pos[11][2] + keypoint_pos[12][2])/2.0};
        
        Eigen::Vector3d temp1((keypoint_pos[6][0]-keypoint_pos[5][0]),(keypoint_pos[6][1]-keypoint_pos[5][1]),(keypoint_pos[6][2]-keypoint_pos[5][2]));
        Eigen::Vector3d temp2((keypoint_pos[12][0]-keypoint_pos[11][0]),(keypoint_pos[12][1]-keypoint_pos[11][1]),(keypoint_pos[12][2]-keypoint_pos[11][2]));        
        // part.radius = (temp1.norm() + temp2.norm()) / 4.0;
    }
    else{
        part.start_p = keypoint_pos[part.joints[0]];
        part.end_p = keypoint_pos[part.joints[1]];
    }
}


bool Human::check_Cylinder(std::vector<double> a, std::vector<double> b, std::vector<double> s, double r){
    double ab = sqrt(pow((a[0] - b[0]), 2.0) + pow((a[1] - b[1]), 2.0) + pow((a[2] - b[2]), 2.0));
    double as = sqrt(pow((a[0] - s[0]), 2.0) + pow((a[1] - s[1]), 2.0) + pow((a[2] - s[2]), 2.0));
    double bs = sqrt(pow((s[0] - b[0]), 2.0) + pow((s[1] - b[1]), 2.0) + pow((s[2] - b[2]), 2.0));
    double cos_A = (pow(as, 2.0) + pow(ab, 2.0) - pow(bs, 2.0)) / (2 * ab*as);
    double sin_A = sqrt(1 - pow(cos_A, 2.0));
    double dis = as*sin_A;

    double inner_sab = (a[0]-b[0])*(a[0]-s[0]) + (a[1]-b[1])*(a[1]-s[1]) + (a[2]-b[2])*(a[2]-s[2]);
    double inner_sba = (b[0]-a[0])*(b[0]-s[0]) + (b[1]-a[1])*(b[1]-s[1]) + (b[2]-a[2])*(b[2]-s[2]);

    return dis < r && inner_sab > 0 && inner_sba > 0;
}

void Human::get_part_cylinder(){
    double spacing = 0.03;

    // 为点设置颜色（这里用红色示例）
    uint8_t r = 255;
    uint8_t g = 0;
    uint8_t b = 0;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

    for(BodyPart& part : human_dict){
        double height;
        switch (part.type){
            case 1: height = 0.4; // 头
            case 2: height = 0.6; // 身体
            case 3: height = 0.5; // 四肢
        }

        int point_index = 0;

        // 计算圆柱上每个点的坐标和颜色并加入点云
        for (double z = -height / 2.0; z <= height / 2.0; z += spacing) {
            // 若为完整圆柱，采用 0 - 2.0 * M_PI
            for (double theta = M_PI; theta < M_PI*2.0; theta += spacing / part.radius) {
                pcl::PointXYZRGB point;
                point.x = part.radius * cos(theta);
                point.z = part.radius * sin(theta);
                point.y = z;

                point.rgb = *reinterpret_cast<float*>(&rgb);

                part.cylinder_model->points.push_back(point);
            }
        }

        part.cylinder_model->height = 1;
        part.cylinder_model->width = part.cylinder_model->points.size();
        ROS_INFO("[%s] Part Cylinder_model PointCloud Size = %i ",part.name.c_str(), part.cylinder_model->width);
    }
}

void Human::get_model(){
    for(BodyPart& part : human_dict){
        if(!part.exist) continue;

        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cylinder(new pcl::PointCloud<pcl::PointXYZRGB>);

        part.trans_cylinder.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Eigen::Vector3f start(part.start_p[0], part.start_p[1], part.start_p[2]);
        // Eigen::Vector3f end(part.end_p[0], part.end_p[1], part.end_p[2]);

        Eigen::Vector3f start(0.0, -1.0, 1.0);
        Eigen::Vector3f end(0.0, 0.0, 1.0);

        // 将start与end交换，半圆柱的朝向也换相反而且错误

        Eigen::Vector3f origin = (start + end) * 0.5;
        Eigen::Vector3f z_direction = (end - start).normalized();

        Eigen::Vector3f x_axis(1.0, 0.0, 0.0);

        Eigen::Vector3f y_direction = x_axis.cross(z_direction).normalized();

        Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();

        // Eigen::Matrix4f deeper_matrix;
        // deeper_matrix << 1.0, 0.0, 0.0, 0.0,
        //                  0.0, 1.0, 0.0, 0.0,
        //                  0.0, 0.0, 1.0, part.radius*0.5,
        //                  0.0, 0.0, 0.0, 1.0;

        transformation_matrix.block<3, 1>(0, 3) = origin;
        transformation_matrix.block<3, 1>(0, 1) = y_direction;
        transformation_matrix.block<3, 1>(0, 2) = z_direction;

        // pcl::transformPointCloud(*part.cylinder_model, *part.trans_cylinder, deeper_matrix * transformation_matrix);

        pcl::transformPointCloud(*part.cylinder_model, *part.trans_cylinder, transformation_matrix);
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
}

/***  CAM_1 Pose处理  ***/
void pose_cb1(const seg_ros::KeypointsWithScores& msg)
{
    human1.keypoints.clear();
    for(const geometry_msgs::Point& point : msg.keypoints){
        std::vector<int> single_point;
        single_point.push_back(static_cast<int>(point.x));
        single_point.push_back(static_cast<int>(point.y));
        
        human1.keypoints.push_back(single_point);
    }

    human1.keypoint_scores.clear();
    for(const std_msgs::Float32& score : msg.keypoint_scores){
        human1.keypoint_scores.push_back(score.data);
    }

    human1.keypoint_pos.clear();
    for(const std::vector<int>& joint : human1.keypoints){       
        std::vector<double> j(3,0.0);
        
        float d = cam1.depth_pic.ptr<float>(joint[1])[joint[0]];
        // 利用图像的link长度求取先验深度而非直接读取！！36

        j[2] = double(d) / cam1.camera_factor;
        j[0] = (joint[0] - cam1.cx) * j[2] / cam1.fx;
        j[1] = (joint[1] - cam1.cy) * j[2] / cam1.fy;

        ROS_INFO("depth: %f",d);
        ROS_INFO("joint: (%i, %i)", joint[0], joint[1]);
        ROS_INFO("joint_pos: (%lf, %lf, %lf)", j[0], j[1], j[2]);

        human1.keypoint_pos.push_back(j);
    }

    // std::cout << "Cam1 Keypoints: " << std::endl;
    // for (const std::vector<double>& inner : human1.keypoints) {
    //     std::cout << "[ ";
    //     for (double value : inner) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << "] ";
    //     std::cout << std::endl;
    // }

    // std::cout << "Cam1 Keypoint Scores: " << std::endl;
    // std::cout << "[ ";
    // for (double value : human1.keypoint_scores) {
    //     std::cout << value << " ";
    // }
    // std::cout << "] ";
    // std::cout << std::endl;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "human_model");
    ros::NodeHandle nh;

    // D435相机的图像话题订阅和相机参数设定
    image_transport::ImageTransport it1(nh);
    image_transport::Subscriber sub1_color = it1.subscribe(("/cam_"+cam1.cam_num+"/color/image_raw"), 1, color_cb1);
    image_transport::Subscriber sub1_depth = it1.subscribe(("/cam_"+cam1.cam_num+"/aligned_depth_to_color/image_raw"), 1, depth_cb1);

    tf::TransformListener* lis_cam1 = new(tf::TransformListener);
    
    cam1 = Camera(1,lis_cam1);

    cam1.fx = 608.7494506835938;
    cam1.fy = 608.6277465820312;
    cam1.cx = 315.4583435058594;
    cam1.cy = 255.28733825683594;

    // 读取python发布的mm pose结果
    ros::Subscriber sub1_pose = nh.subscribe(("cam_"+cam1.cam_num+"/pose"),1,pose_cb1);
    
    // mm pose结果和深度图像的时间戳对齐
    
    // 人模型的维护
    BodyPart head("head", 1, {0,1,2,3,4,5,6}, 0.1);
    BodyPart body("body", 2, {5,6,11,12}, 0.2);

    BodyPart arm_left_upper("arm_left_upper", 3, {5,7}, 0.06);
    BodyPart arm_left_lower("arm_left_lower", 3,{7,9}, 0.04);
    BodyPart arm_right_upper("arm_right_upper", 3,{6,8}, 0.06);
    BodyPart arm_right_lower("arm_right_lower", 3,{8,10}, 0.04);
    
    BodyPart leg_left_upper("leg_left_upper", 3,{11,13}, 0.08);
    BodyPart leg_left_lower("leg_left_lower", 3,{13,15}, 0.06);
    BodyPart leg_right_upper("leg_right_upper", 3,{12,14}, 0.08);
    BodyPart leg_right_lower("leg_right_lower", 3,{14,16}, 0.06);

    human1.add_part(head);
    human1.add_part(body);
    human1.add_part(arm_left_upper);
    human1.add_part(arm_left_lower);
    human1.add_part(arm_right_upper);
    human1.add_part(arm_right_lower);
    human1.add_part(leg_left_upper);
    human1.add_part(leg_left_lower);
    human1.add_part(leg_right_upper);
    human1.add_part(leg_right_lower);

    human1.get_part_cylinder();

    // 由joint位置作为先验生成几何包络

    // 发布人模型的几何表征结果

    ros::Rate loop_rate(30.0);

    sensor_msgs::PointCloud2 msg_human, msg_part;

    ros::Publisher pub_human = nh.advertise<sensor_msgs::PointCloud2>("/pc_human", 1);
    ros::Publisher pub_part = nh.advertise<sensor_msgs::PointCloud2>("/pc_part", 1);

    while(ros::ok())
    {
        cam1.pic2cloud(); // 生成人的点云
        
        human1.seg_cloud(cam1); // 分割生成存在部位点云

        human1.get_model(); // 获取人的圆柱模型表征

        pcl::toROSMsg(*cam1.cam_pc, msg_human);
        msg_human.header.frame_id = "cam_1_link";
        msg_human.header.stamp = ros::Time::now();
        pub_human.publish(msg_human); // 发布整体人的点云

        pcl::toROSMsg(*human1.human_dict[1].point_cloud, msg_part);
        msg_part.header.frame_id = "cam_1_link";
        // pcl::toROSMsg(*human1.human_dict[1].trans_cylinder, msg_part);
        // msg_part.header.frame_id = "cam_1_link";
        msg_part.header.stamp = ros::Time::now();
        pub_part.publish(msg_part); // 发布提取部位点云
        
        ros::spinOnce();
        loop_rate.sleep();
    }
}