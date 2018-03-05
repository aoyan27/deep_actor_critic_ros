#!/usr/bin/env python
#coding:utf-8
import argparse

import rospy

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

from std_msgs.msg import Float64, Int64, Float64MultiArray, Bool
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

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

class pendulum:
    def __init__(self, limit_simulation_time, initial_theta, initial_omega):
        self.Loop_Rate = 100
        
        #  self.mu = 0.0
        self.mu = 0.01
        #  self.mu = 0.7
        self.gravity = 9.81

        self.limit_sim_time = limit_simulation_time
        #  self.dt = 1.0 / self.Loop_Rate
        self.dt = 0.0001

        self.init_theta = initial_theta
        self.init_omega = initial_omega
        
        self.torque = 0.0

        self.numpy_time = np.array([])
        self.numpy_theta = np.array([])
        self.numpy_omega = np.array([])
        self.numpy_torque = np.array([])

        self.limit_iteration = 10000000
        
        self.episode_finish_flag = False

    def episode_finish_callback(self, msg):
        self.episode_finish_flag = msg.data
    
    def torque_callback(self, msg):
        self.torque = msg.data

    def function1(self, Theta, Omega, Torque, t):
        return Omega

    def function2(self, Theta, Omega, Torque, t):
        #  return (-self.mu*Omega + self.gravity*math.sin(Theta) + Torque)
        return (-1*Theta)
    
    def euler(self, Theta, Omega, Torque, Time):
        k_theta = self.function1(Theta, Omega, Torque, Time)
        k_omega = self.function2(Theta, Omega, Torque, Time)

        Theta += k_theta * self.dt
        Omega += k_omega * self.dt

        return Theta, Omega

    
    def runge_kutta(self, Theta, Omega, Torque, Time):
        k1 = np.zeros((1, 2), dtype=np.float32)
        k2 = np.zeros((1, 2), dtype=np.float32)
        k3 = np.zeros((1, 2), dtype=np.float32)
        k4 = np.zeros((1, 2), dtype=np.float32)
        
        k1[0][0] = self.dt * self.function1(Theta, Omega, Torque, Time)
        k1[0][1] = self.dt * self.function2(Theta, Omega, Torque, Time)

        k2[0][0] = self.dt * self.function1(Theta+k1[0][0]/2.0, Omega+k1[0][1]/2.0, Torque, Time+self.dt/2.0)
        k2[0][1] = self.dt * self.function2(Theta+k1[0][0]/2.0, Omega+k1[0][1]/2.0, Torque, Time+self.dt/2.0)

        k3[0][0] = self.dt * self.function1(Theta+k2[0][0]/2.0, Omega+k2[0][1]/2.0, Torque, Time+self.dt/2.0)
        k3[0][1] = self.dt * self.function2(Theta+k2[0][0]/2.0, Omega+k2[0][1]/2.0, Torque, Time+self.dt/2.0)

        k4[0][0] = self.dt * self.function1(Theta+k3[0][0]/2.0, Omega+k3[0][1]/2.0, Torque, Time+self.dt)
        k4[0][1] = self.dt * self.function2(Theta+k3[0][0]/2.0, Omega+k3[0][1]/2.0, Torque, Time+self.dt)
        
        Theta = Theta + (k1[0][0] + 2.0*k2[0][0] + 2.0*k3[0][0] + k4[0][0]) / 6.0
        Omega = Omega + (k1[0][1] + 2.0*k2[0][1] + 2.0*k3[0][1] + k4[0][1]) / 6.0

        return Theta, Omega


    def main(self):
        rospy.init_node('sumulation_simple_pendulum')

        rospy.Subscriber("/torque", Float64, self.torque_callback)
        rospy.Subscriber("/episode_finish", Bool, self.episode_finish_callback)

        pub_time = rospy.Publisher("/time", Float64, queue_size = 1)
        pub_theta = rospy.Publisher("/theta", Float64, queue_size = 1)
        pub_omega = rospy.Publisher("/omega", Float64, queue_size = 1)

        time = 0.0
        theta = self.init_theta
        omega = self.init_omega

        i = 0

        loop_rate = rospy.Rate(self.Loop_Rate)



        #  while not rospy.is_shutdown():
        while 1:
            

            if time <= 5:
                self.torque = 0.0
            elif 5<time and time<=10:
                self.torque = 0.0
            elif 10<time<=15:
                self.torque = 0.0
            elif 15<time<=20:
                self.torque = 0.0
            elif 20<time<=25:
                self.torque = 0.0
            else:
                self.torque = 0.0

            time = self.dt * i

            #  theta, omega = self.euler(theta, omega, self.torque, time)
            theta, omega = self.runge_kutta(theta, omega, self.torque, time)
            
            if self.episode_finish_flag:
            #  if time == 16.0:
                time = 0.0
                i = 0
                self.show_figure(time, theta, omega, self.torque, True)
                pub_time.publish(time)
                pub_theta.publish(theta)
                pub_omega.publish(omega)

                theta = self.init_theta
                omega = self.init_omega
                
                self.numpy_time = np.array([])
                self.numpy_theta = np.array([])
                self.numpy_omega = np.array([])
                self.numpy_torque = np.array([])
                print "Simulation Finish!!"
                break
            else:
                print "time : %f / torque : %f / theta : %f / omega : %f" % (time, self.torque, theta, omega)        
                #  print "time : %f / torque : %f / theta : %f / omega : %f" % (time, self.torque, math.degrees(theta), omega)        
                self.show_figure(time, theta, omega, self.torque)
                pub_time.publish(time)
                pub_theta.publish(theta)
                pub_omega.publish(omega)
            
            i += 1

            #  loop_rate.sleep()

        #  self.show_figure(time, theta, omega, self.torque, True)

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

if __name__=="__main__":
    limit_time = 60
    init_theta = 1.0
    #  init_theta = 90.0
    init_omega = 0.0
    torque = 0.0

    simulation_simple_pendulum = pendulum(limit_time,init_theta, init_omega)
    #  simulation_simple_pendulum = pendulum(limit_time,math.radians(init_theta), init_omega)
    #  simulation_simple_pendulum.simulation(torque)
    simulation_simple_pendulum.main()
