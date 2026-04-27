//serial_imu.cpp
#include <ros/ros.h>
#include <serial/serial.h>
#include <sensor_msgs/Imu.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#ifdef __cplusplus 
extern "C"{
#endif

#include <stdint.h>
#include <stdbool.h>

#include "ch_serial.h"
///dev/ttyUSB0  /dev/ttyimu
//115200  921600
#define IMU_SERIAL   "/dev/ttyimu"
#define BAUD         (921600)
#define GRA_ACC      (9.8)
#define DEG_TO_RAD   (0.01745329)
#define BUF_SIZE     1024
#define TIMER_PERIOD (0.0025)

void publish_imu_data(raw_t *data, sensor_msgs::Imu *imu_data);
void load_covariance_param(ros::NodeHandle *private_node,
                           const std::string &param_name,
                           const std::vector<double> &default_values,
                           boost::array<double, 9> *target);

#ifdef __cplusplus
}
#endif

static raw_t raw;
ros::Publisher IMU_pub;
serial::Serial sp;
sensor_msgs::Imu imu_data;
std::string imu_frame_id = "base_link";

void callback(const ros::TimerEvent& event)
{
	int rev = 0;
	size_t num = sp.available();
	if(num!=0)
	{
		uint8_t buffer[BUF_SIZE]; 

		if(num > BUF_SIZE)
			num = BUF_SIZE;

		num = sp.read(buffer, num);
		if(num > 0)
		{
			imu_data.header.frame_id = imu_frame_id;

			for (int i = 0; i < num; i++)
			{
				rev = ch_serial_input(&raw, buffer[i]);

				if(raw.item_code[raw.nitem_code - 1] != KItemGWSOL)
				{
					if(rev)
					{
						imu_data.header.stamp = ros::Time::now();
						publish_imu_data(&raw, &imu_data);
						IMU_pub.publish(imu_data);
					}
				}
			}
		}
	}
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "serial_port");
	ros::NodeHandle n;
	ros::NodeHandle private_node("~");

	std::string serial_port = IMU_SERIAL;
	int baud = BAUD;
	double timer_period = TIMER_PERIOD;

	private_node.param<std::string>("serial_port", serial_port, IMU_SERIAL);
	private_node.param<int>("baud", baud, BAUD);
	private_node.param<std::string>("frame_id", imu_frame_id, "base_link");
	private_node.param<double>("timer_period", timer_period, TIMER_PERIOD);

	load_covariance_param(
		&private_node,
		"orientation_covariance",
		std::vector<double>{-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		&imu_data.orientation_covariance);
	load_covariance_param(
		&private_node,
		"angular_velocity_covariance",
		std::vector<double>{1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4},
		&imu_data.angular_velocity_covariance);
	load_covariance_param(
		&private_node,
		"linear_acceleration_covariance",
		std::vector<double>{1e-2, 0.0, 0.0, 0.0, 1e-2, 0.0, 0.0, 0.0, 1e-2},
		&imu_data.linear_acceleration_covariance);

    IMU_pub = n.advertise<sensor_msgs::Imu>("/IMU_data", 20);

	serial::Timeout to = serial::Timeout::simpleTimeout(100);

	sp.setPort(serial_port);

	sp.setBaudrate(baud);

	sp.setTimeout(to);
	

	try
	{
		sp.open();
	}
	catch(serial::IOException& e)
	{
		ROS_ERROR_STREAM("Unable to open port.");
		return -1;
	}
    
	if(sp.isOpen())
	{
		ROS_INFO_STREAM(serial_port << " is opened.");
		ROS_INFO_STREAM("IMU frame_id=" << imu_frame_id
			<< ", timer_period=" << timer_period);
	}
	else
	{
		return -1;
	}
	
	ros::Timer timer = n.createTimer(ros::Duration(timer_period), callback);

	ros::spin();

	sp.close();
 
	return 0;
}

void load_covariance_param(ros::NodeHandle *private_node,
                           const std::string &param_name,
                           const std::vector<double> &default_values,
                           boost::array<double, 9> *target)
{
	std::vector<double> values;
	if (!private_node->getParam(param_name, values))
	{
		values = default_values;
	}

	if (values.size() != target->size())
	{
		ROS_WARN_STREAM(
			"Parameter " << private_node->getNamespace() << "/" << param_name
			<< " must contain exactly " << target->size()
			<< " values. Falling back to defaults.");
		values = default_values;
	}

	for (std::size_t i = 0; i < target->size(); ++i)
	{
		(*target)[i] = values[i];
	}
}

void publish_imu_data(raw_t *data, sensor_msgs::Imu *imu_data)
{	
	imu_data->orientation.x = data->imu[data->nimu - 1].quat[1];
	imu_data->orientation.y = data->imu[data->nimu - 1].quat[2];
	imu_data->orientation.z = data->imu[data->nimu - 1].quat[3];
	imu_data->orientation.w = data->imu[data->nimu - 1].quat[0];
	imu_data->angular_velocity.x = data->imu[data->nimu - 1].gyr[0] * DEG_TO_RAD;
	imu_data->angular_velocity.y = data->imu[data->nimu - 1].gyr[1] * DEG_TO_RAD;
	imu_data->angular_velocity.z = data->imu[data->nimu - 1].gyr[2] * DEG_TO_RAD;
	imu_data->linear_acceleration.x = data->imu[data->nimu - 1].acc[0] * GRA_ACC;
	imu_data->linear_acceleration.y = data->imu[data->nimu - 1].acc[1] * GRA_ACC;
	imu_data->linear_acceleration.z = data->imu[data->nimu - 1].acc[2] * GRA_ACC;
}


