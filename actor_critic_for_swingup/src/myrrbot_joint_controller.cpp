#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int64.h>
#include <stdio.h>
using namespace std;

double pi = 3.141592;

std_msgs::Float64 joint1_command;

bool joint1_command_flag = false;

void joint1_command_callback(std_msgs::Float64 msg){
	joint1_command_flag = true;
	joint1_command.data = msg.data;
	cout<<"Subscribe joint1 command : "<<joint1_command.data<<endl;
}


int main(int argc, char *argv[]){
	ros::init(argc, argv, "myrrbot_joint_controller");
	ros::NodeHandle n;

	ros::Subscriber sub = n.subscribe("/action", 10, joint1_command_callback);

	ros::Publisher pub = n.advertise<std_msgs::Float64>("/my_rrbot/joint1_controller/command", 1);

	ros::Rate loop_rate(50);
	

	while(ros::ok()){
		if(joint1_command_flag){
			pub.publish(joint1_command);
			printf("Publish joint1 command!!\n");
			joint1_command_flag = false;
		}

		ros::spinOnce();
		loop_rate.sleep();
	}
	return 0;
}
