#include <ros/ros.h>
#include <std_msgs/Float64.h>

using namespace std;

double pi = 3.141592;

int main(int argc, char *argv[]){
	ros::init(argc, argv, "sample_publisher");
	ros::NodeHandle n;

	ros::Publisher pub_1 = n.advertise<std_msgs::Float64>("/action", 1);
	ros::Rate loop_rate(50);
	
	std_msgs::Float64 joint1_com;

	int count = 0;
	while(ros::ok()){
		count += 1;
		if(count <= 100000){
			cout<<"count : "<<count<<endl;
			if((count%10)==0){
				joint1_com.data = 5.0;
			}
			else{
				joint1_com.data = 5.0;
			}
		}
		else{
			if((count%2)==0){
				joint1_com.data = 5.0;
			}
			else{
				joint1_com.data = 5.0;
			}
		}
		cout<<"publish"<<endl;
		pub_1.publish(joint1_com);

		printf("joint1 : %.1f\n", joint1_com.data);

		loop_rate.sleep();
	}
	return 0;
}
