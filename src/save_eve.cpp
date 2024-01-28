#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <seg_ros/KeypointsWithScores.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>

int bag_num = 30;

// 定义全局变量用于保存图像和点云数据
cv::Mat cam1_color;
// cv::Mat cam1_depth;
cv::Mat cam2_color;
// cv::Mat cam2_depth;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam1_pc(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cam2_pc(new pcl::PointCloud<pcl::PointXYZRGB>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr human_pc(new pcl::PointCloud<pcl::PointXYZRGB>());
Eigen::VectorXd cam1_score(17);
Eigen::VectorXd cam2_score(17);
double curr_time;

bool color1_flag = false;
bool color2_flag = false;
bool depth1_flag = true;
bool depth2_flag = true;
bool cloud1_flag = false;
bool cloud2_flag = false;
bool cloud3_flag = false;
bool pose1_flag = false;
bool pose2_flag = false;

std::string root_dir = "/home/joy/Documents/24-0122/exist/active/data";

std::string color1_dir = root_dir + "/color1/";
std::string color2_dir = root_dir + "/color2/";
// std::string depth1_dir = "/home/joy/Documents/24-0122/mask/data/depth1/";
// std::string depth2_dir = "/home/joy/Documents/24-0122/mask/data/depth2/";
std::string cloud1_dir = root_dir + "/cloud1/";
std::string cloud2_dir = root_dir + "/cloud2/";
std::string cloud3_dir = root_dir + "/cloud3/";
std::string pose_filename = root_dir + "/pose/bag" + std::to_string(bag_num) + ".txt";

int file_count = 1;

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

    cam1_color = color_ptr->image;
    color1_flag = true;
}

// /***  CAM1 Depth处理  ***/
// void depth_cb1(const sensor_msgs::ImageConstPtr& depth_msg)
// {
//     cv_bridge::CvImagePtr depth_ptr;
//     try
//     {
//         depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
//         cv::waitKey(50);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
//     }
// 
//     cam1_depth = depth_ptr->image;
//     depth1_flag = true;
// }

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

    cam2_color = color_ptr->image;
    color2_flag = true;
}

// /***  CAM2 Depth处理  ***/
// void depth_cb2(const sensor_msgs::ImageConstPtr& depth_msg)
// {
//     cv_bridge::CvImagePtr depth_ptr;
//     try
//     {
//         depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
//         cv::waitKey(50);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("Could not convert from '%s' to 'mono16'.", depth_msg->encoding.c_str());
//     }
//  
//     cam2_depth = depth_ptr->image;
//     depth2_flag = true;
// }

void pc_cb1(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // 将ROS点云消息转换为PCL点云类型
    pcl::fromROSMsg(*msg, *cam1_pc);
    cloud1_flag = true;
}

/***  CAM2 点云处理  ***/
void pc_cb2(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // 将ROS点云消息转换为PCL点云类型
    pcl::fromROSMsg(*msg, *cam2_pc);
    cloud2_flag = true;
}

void pc_cb3(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // 将ROS点云消息转换为PCL点云类型
    pcl::fromROSMsg(*msg, *human_pc);
    cloud3_flag = true;
}

/***  CAM_1 Pose处理  ***/
void pose_cb1(const seg_ros::KeypointsWithScores& msg)
{
    for(int i=0; i<17; i++){
        cam1_score(i) = msg.keypoint_scores[i].data;
    }

    pose1_flag = true;
}

/***  CAM_2 Pose处理  ***/
void pose_cb2(const seg_ros::KeypointsWithScores& msg)
{
    curr_time = msg.header.stamp.sec + msg.header.stamp.nsec / 1000000000.0;

    for(int i=0; i<17; i++){
        cam2_score(i) = msg.keypoint_scores[i].data;
    }

    pose2_flag = true;
}

// 定时器回调函数：每隔0.5秒保存图像和点云数据
void timer_cb(const ros::TimerEvent& event) {
    if(color1_flag && color2_flag && depth1_flag && depth2_flag && cloud1_flag && cloud2_flag && pose1_flag && pose2_flag){
        if(!cam1_color.empty() && !cam2_color.empty()){
            ROS_INFO("Recording file %i", file_count);

            std::string color1_filename = color1_dir + "color"+ std::to_string(bag_num) +"_" + std::to_string(file_count) + ".png";
            std::string color2_filename = color2_dir + "color"+ std::to_string(bag_num) +"_" + std::to_string(file_count) + ".png";
            // std::string depth1_filename = depth1_dir + "depth_" + std::to_string(file_count) + ".png";
            // std::string depth2_filename = depth2_dir + "depth_" + std::to_string(file_count) + ".png";
            std::string cloud1_filename = cloud1_dir + "cloud"+ std::to_string(bag_num) +"_" + std::to_string(file_count) + ".pcd";
            std::string cloud2_filename = cloud2_dir + "cloud"+ std::to_string(bag_num) +"_" + std::to_string(file_count) + ".pcd";
            std::string cloud3_filename = cloud3_dir + "cloud"+ std::to_string(bag_num) +"_" + std::to_string(file_count) + ".pcd";

            cv::imwrite(color1_filename, cam1_color);
            cv::imwrite(color2_filename, cam2_color);
            // cv::imwrite(depth1_filename, cam1_depth);
            // cv::imwrite(depth2_filename, cam2_depth);

            if(!cam1_pc->empty()){
                pcl::io::savePCDFileBinary(cloud1_filename, *cam1_pc);
            }
            
            if(!cam2_pc->empty()){
                pcl::io::savePCDFileBinary(cloud2_filename, *cam2_pc);
            }

            if(!human_pc->empty()){
                pcl::io::savePCDFileBinary(cloud3_filename, *human_pc);
            }

            std::ofstream fileHandle(pose_filename, std::ios::app);

            fileHandle << curr_time << "    ";

            for (int i = 0; i < 17; ++i) {
                fileHandle << cam1_score(i) << "    ";
            }

            for (int i = 0; i < 17; ++i) {
                fileHandle << cam2_score(i) << "    ";
            }

            fileHandle << "\n";

            fileHandle.close();

            color1_flag = false;
            color2_flag = false;
            // depth1_flag = false;
            // depth2_flag = false;
            cloud1_flag = false;
            cloud2_flag = false;
            cloud3_flag = false;
            pose1_flag = false;
            pose2_flag = false;

            file_count++;
        }
        else{
            ROS_INFO("Messages have no data!");
        }
    }
    else{
        ROS_INFO("Waiting for messages...");
    }
}

void clearDirectory(const std::string& directoryPath) {
    try {
        boost::filesystem::path directory(directoryPath);
        boost::filesystem::remove_all(directory);
        ROS_INFO("Cleared directory: %s", directoryPath.c_str());
    } catch (const boost::filesystem::filesystem_error& e) {
        ROS_ERROR("Failed to clear directory: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "save_eve");
    ros::NodeHandle nh;

    // 创建图像传输对象
    image_transport::ImageTransport it(nh);

    // 订阅图像话题
    image_transport::Subscriber colorSub1 = it.subscribe("/cam_1/color/image_raw", 1, color_cb1);
    // image_transport::Subscriber depthSub1 = it.subscribe("/cam_1/aligned_depth_to_color/image_raw", 1, depth_cb1);
    image_transport::Subscriber colorSub2 = it.subscribe("/cam_2/color/image_raw", 1, color_cb2);
    // image_transport::Subscriber depthSub2 = it.subscribe("/cam_2/aligned_depth_to_color/image_raw", 1, depth_cb2);

    // 订阅点云话题
    ros::Subscriber cloudSub1 = nh.subscribe("/pc_1", 1, pc_cb1);
    ros::Subscriber cloudSub2 = nh.subscribe("/pc_2", 1, pc_cb2);
    ros::Subscriber cloudSub3 = nh.subscribe("/pc_human", 1, pc_cb3);
    ros::Subscriber sub1_pose = nh.subscribe(("cam_1/pose"),1,pose_cb1);
    ros::Subscriber sub2_pose = nh.subscribe(("cam_2/pose"),1,pose_cb2);

    // 创建定时器，每隔0.25秒调用一次回调函数
    ros::Timer timer = nh.createTimer(ros::Duration(0.25), timer_cb);

    std::ofstream filePose(pose_filename, std::ios::trunc);
    filePose.close();

    ros::spin();

    return 0;
}
