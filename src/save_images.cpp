#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <string>

int main(int argc, char** argv){
    ros::init(argc, argv, "save_images");
    ros::NodeHandle nh;

    rosbag::Bag bag;
    bag.open("/home/joy/Documents/new_bag/1.bag", rosbag::bagmode::Read);

    std::vector<std::string> topics;

    topics.push_back("/camera/color/image_raw");
    topics.push_back("/camera/aligned_depth_to_color/image_raw");

    std::string color_dir = "/home/joy/mm_ws/src/seg_ros/images/color/";
    std::string depth_dir = "/home/joy/mm_ws/src/seg_ros/images/depth/";

    int color_count = 0, depth_count = 0;
    ros::Time last_color_time, last_depth_time;

    for(rosbag::MessageInstance const& msg : rosbag::View(bag, rosbag::TopicQuery(topics))){
        if(color_count==0 && depth_count==0){
            last_color_time = msg.getTime();
            last_depth_time = msg.getTime();
            color_count++;
            depth_count++;
        }
        
        ros::Time msg_time = msg.getTime();

        // 深度图像
        if(msg.getTopic()=="/camera/aligned_depth_to_color/image_raw" && (msg_time-last_depth_time).toSec() >= 1.0){
            sensor_msgs::Image::ConstPtr image_msg = msg.instantiate<sensor_msgs::Image>();
            if (image_msg != nullptr){
                cv_bridge::CvImageConstPtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(image_msg, image_msg->encoding);
                } catch (cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    continue;
                }
                std::string depth_image_filename = depth_dir + "depth_" + std::to_string(depth_count) + ".png";
                cv::imwrite(depth_image_filename, cv_ptr->image);
                ROS_INFO("Saved depth image to %s", depth_image_filename.c_str());

                last_depth_time = msg_time;
                depth_count++;
            }
        }

        // 彩色图像
        if(msg.getTopic()=="/camera/color/image_raw" && (msg_time-last_color_time).toSec() >= 1.0){
            sensor_msgs::Image::ConstPtr image_msg = msg.instantiate<sensor_msgs::Image>();
            if (image_msg != nullptr){
                cv_bridge::CvImageConstPtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
                } catch (cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    continue;
                }
                std::string color_image_filename = color_dir + "color_" + std::to_string(color_count) + ".png";
                cv::imwrite(color_image_filename, cv_ptr->image);
                ROS_INFO("Saved color image to %s", color_image_filename.c_str());

                last_color_time = msg_time;
                color_count++;
            }
        }

    }

    bag.close();

    return 0;
}