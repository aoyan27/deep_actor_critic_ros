#!/usr/bin/env python
#coding:utf-8
import argparse

import rospy

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

from std_msgs.msg import Float64, Int64, Float64MultiArray
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from sensor_msgs.msg import JointState

import tf

import math
from random import random, randint, uniform

import pickle

import matplotlib.pyplot as plt

 
parser = argparse.ArgumentParser(description='deep_actor_critic_for_swingup')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >=0 else np

class test_gazebo:
    def __init__(self):
        self.numpy_time = np.array([])
        self.numpy_torque = np.array([])
        self.numpy_theta = np.array([])
        self.numpy_omega = np.array([])

        self.theta = 0.0
        self.omega = 0.0
        self.torque = 0.0

        self.init_theta = math.pi / 2.0

        self.dt = 1.0/ 50.0

        self.flag = False

    def reset_client(self, angle):
        q = tf.transformations.quaternion_from_euler(0.0, angle, 0.0)
        request = LinkState()
        request.link_name = "my_rrbot::link2"
        request.pose.position.x = 0.0
        request.pose.position.y = 0.1
        request.pose.position.z = 1.45
        request.pose.orientation.x = q[0]
        request.pose.orientation.y = q[1]
        request.pose.orientation.z = q[2]
        request.pose.orientation.w = q[3]
        #  print "request.pose : "
        #  print request.pose
        try:
            reset = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
            resp = reset(request)
            #  print "resp : ", resp
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e
    
    def joint_states_callback(self, msg):
        print "Subscribe!!!"
        self.theta = msg.position[0]
        self.omega = msg.velocity[0]
        print "self.theta - self.init_theta : ", self.theta - self.init_theta
        if self.theta-self.init_theta < 1e-2:
            self.flag = True

    def show_figure(self, time, theta, omega, torque, simulation_finish=False): 
        if not simulation_finish:
            self.numpy_time = np.append(self.numpy_time, [time])
            self.numpy_theta = np.append(self.numpy_theta, [theta/math.pi *180.0])
            self.numpy_omega = np.append(self.numpy_omega, [omega])
            self.numpy_torque = np.append(self.numpy_torque, [torque])
            #  print "self.numpy_torque : ", self.numpy_torque
        else:
            fig, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(self.numpy_time, self.numpy_theta)
            axarr[0].set_ylabel('Theta')
            #  axarr[0].set_xlabel('Time')
            axarr[0].grid()
            axarr[0].set_title('Time-Theta')
            axarr[1].plot(self.numpy_time, self.numpy_omega)
            axarr[1].set_ylabel('Omega')
            #  axarr[1].set_xlabel('Time')
            axarr[1].grid()
            axarr[1].set_title('Time-Omega')
            axarr[2].plot(self.numpy_time, self.numpy_torque)
            axarr[2].set_ylabel('Torque')
            #  axarr[2].set_xlabel('Time')
            axarr[2].grid()
            axarr[2].set_title('Time-Torque')
            
            plt.show()

    def main(self):
        rospy.init_node('test_gazebo')

        rospy.Subscriber("/my_rrbot/joint_states", JointState, self.joint_states_callback)

        pub = rospy.Publisher("/my_rrbot/joint1_controller/command", Float64, queue_size=1)

        loop_rate = rospy.Rate(50)

        time = 0.0
        i = 0

        init_theta = math.pi / 2.0

        count = 0

        flag = False

        while not rospy.is_shutdown():
            if count == 0:
                self.reset_client(init_theta)
            print "self.theta : ", self.theta
            if self.flag:
                print "start!!!!"
                print "count : ", count
                time = i*self.dt

                pub.publish(self.torque)
                self.show_figure(time, self.theta, self.omega, self.torque)

                if count >= 800:
                    self.show_figure(time, self.theta, self.omega, self.torque, True)
                    break
                
                i += 1
                count += 1
            loop_rate.sleep()

    
if __name__=="__main__":
    test = test_gazebo()
    test.main()
