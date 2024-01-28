#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tf_sim");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10.0);
    
    tf::TransformBroadcaster tf_broadcaster;

    Eigen::Matrix4d cam1_to_world, cam2_to_world, cam2_to_cam1;
    tf::StampedTransform cam1_trans, cam2_trans;

    // cam1_to_world<< 0.587395, -0.143131, -0.796543, -0.0447751,
    //                 0.8093, 0.103886, 0.578136, -0.402986,
    //                 0, -0.984236, 0.176858, -0.760546,
    //                 0, 0, 0, 1;
    
    // cam2_to_world<< -0.587395, -0.143131, -0.796543, -0.0447751,
    //                 0.8093, -0.103886, -0.578136, 0.402986,
    //                 0, -0.984236, 0.176858, -0.760546,
    //                 0, 0, 0, 1;

    // // 23-1206数据包
    // cam1_to_world<<       
    //      0.5, -0.433013,     -0.75, -0.263071,
    // 0.866025,      0.25,  0.433013, -0.363586,
    //        0, -0.866025,       0.5,    -0.725,
    //        0,         0,         0,         1;
    
    // cam2_to_cam1<<
    // 0.540077,    -0.346834,    -0.766827,   0.62877,
    // 0.360381,     0.918683,    -0.161701,  0.160256,
    // 0.760554,    -0.189019,     0.621151,  0.313506,
    //        0,            0,            0,         1;

    // 24-0112数据包（cam1_to_world乱给的）
    cam1_to_world<<       
         0.5, -0.433013,     -0.75, -0.263071,
    0.866025,      0.25,  0.433013, -0.363586,
           0, -0.866025,       0.5,    -0.725,
           0,         0,         0,         1;
    
    cam2_to_cam1<<
    0.662813,  -0.364718,  -0.653956,   0.283919,
    0.285435,    0.93048,  -0.229637,   0.222002,
    0.692246, -0.0344554,   0.720838,   0.448376,
    0,          0,          0,          1;

    cam2_to_world = cam1_to_world * cam2_to_cam1;

    Eigen::Vector3d cam1_pos = cam1_to_world.block<3,1>(0,3);
    Eigen::Quaterniond cam1_quat = Eigen::Quaterniond(cam1_to_world.block<3,3>(0,0));
    cam1_trans.setOrigin(tf::Vector3(cam1_pos(0),cam1_pos(1),cam1_pos(2)));
    cam1_trans.setRotation(tf::Quaternion(cam1_quat.x(),cam1_quat.y(),cam1_quat.z(),cam1_quat.w()));

    Eigen::Vector3d cam2_pos = cam2_to_world.block<3,1>(0,3);
    Eigen::Quaterniond cam2_quat = Eigen::Quaterniond(cam2_to_world.block<3,3>(0,0));
    cam2_trans.setOrigin(tf::Vector3(cam2_pos(0),cam2_pos(1),cam2_pos(2)));
    cam2_trans.setRotation(tf::Quaternion(cam2_quat.x(),cam2_quat.y(),cam2_quat.z(),cam2_quat.w()));

    while (ros::ok())
    {
        // tf_broadcaster.sendTransform(tf::StampedTransform(cam1_trans,ros::Time::now(),"base_link", "cam_1_link"));
        // tf_broadcaster.sendTransform(tf::StampedTransform(cam2_trans,ros::Time::now(),"base_link", "cam_2_link"));
        tf_broadcaster.sendTransform(tf::StampedTransform(cam1_trans,ros::Time::now(),"world", "cam_1_link"));
        tf_broadcaster.sendTransform(tf::StampedTransform(cam2_trans,ros::Time::now(),"world", "cam_2_link"));

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
