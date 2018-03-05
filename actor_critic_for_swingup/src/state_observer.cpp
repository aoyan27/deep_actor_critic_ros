#include "ros/ros.h"
#include <std_msgs/Float64.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <math.h>
#include <stdio.h>


using namespace std;

double pi = 3.141592;

sensor_msgs::JointState joint_states;

std_msgs::Float64MultiArray state;

bool joint_states_flag = false;

void joint_states_callback(sensor_msgs::JointState msg){
	joint_states_flag = true;
	joint_states = msg;
	joint_states.position[0] = fmod(joint_states.position[0], 6*pi);
		
	// cout<<msg<<endl;
	// cout<<"Subscribe joint states position : "<<joint_states.position[0]<<endl;
	// cout<<"Subscribe joint states velocity : "<<joint_states.velocity[0]<<endl;
}


int main(int argc, char *argv[]){
	ros::init(argc, argv, "state_observer");
	ros::NodeHandle n;

	ros::Subscriber sub = n.subscribe("/my_rrbot/joint_states", 10, joint_states_callback);

	ros::Publisher pub = n.advertise<std_msgs::Float64MultiArray>("/state", 1);
	
	ros::Publisher pub_1 = n.advertise<std_msgs::Float64>("/joint_effort", 1);


	ros::Rate loop_rate(50);

	std_msgs::Float64 effort;

	while(ros::ok()){
		if(joint_states_flag){
			joint_states_flag = false;
			state.data.clear();
			state.data.push_back(joint_states.position[0]);
			state.data.push_back(joint_states.velocity[0]);
			
			effort.data = joint_states.effort[0];

			pub.publish(state);
			pub_1.publish(effort);
			cout<<"Subscribe joint states position : "<<joint_states.position[0]<<"("<<joint_states.position[0]/pi*180.0<<")"<<endl;
			cout<<"Subscribe joint states velocity : "<<joint_states.velocity[0]<<endl;
			cout<<"Publish state(position) : "<<state.data[0]<<endl;
			cout<<"Publish state(velocity) : "<<state.data[1]<<endl;
			cout<<"Publish joint effort : "<<effort.data<<endl;
		}

		ros::spinOnce();
		loop_rate.sleep();
	}
	return 0;
}
