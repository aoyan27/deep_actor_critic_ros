#!/usr/bin/env python
#coding:utf-8
import argparse

import rospy

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

from std_msgs.msg import Float64, Int64, Float64MultiArray, String
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

import tf

import math
from random import random, randint, uniform

import pickle
 
parser = argparse.ArgumentParser(description='deep_actor_critic_for_swingup')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >=0 else np

class agent:
    pi = 3.141592
    
    ALPHA = 1e-6
    #  ALPHA = 0.01
    GAMMA = 0.9

    wait_flag = True
    

    filename_critic_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/model_using/myrrbot_critic_model.dat"
    filename_actor_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/model_using/myrrbot_actor_model.dat"

    #  filename_result = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/results/myrrbot6_result.txt"

    def __init__(self):
        self.init_theta = math.pi
        self.init_omega = 0.0

        self.state = xp.array([[self.init_theta, self.init_omega]], dtype=np.float32)
        self.next_state = xp.array([[self.init_theta, self.init_omega]], dtype=np.float32)
        self.reward = 0.0
        self.action = 0.0

        f_actor = open(self.filename_actor_model, 'rb')
        f_critic = open(self.filename_critic_model, 'rb')
        
        self.actor_model = pickle.load(f_actor)
        self.critic_model = pickle.load(f_critic)
        
        """
        self.critic_model = chainer.FunctionSet(
                l1 = F.Linear(2, 1000),
                l2 = F.Linear(1000, 500),
                l3 = F.Linear(500, 250),
                l4 = F.Linear(250, 125),
                l5 = F.Linear(125, 60),
                l6 = F.Linear(60, 30),
                l7 = F.Linear(30, 1, initialW=np.zeros((1, 30), dtype=np.float32)),
                )
        self.actor_model = chainer.FunctionSet(
                l1 = F.Linear(2, 1000),
                l2 = F.Linear(1000, 500),
                l3 = F.Linear(500, 250),
                l4 = F.Linear(250, 125),
                l5 = F.Linear(125, 60),
                l6 = F.Linear(60, 30),
                l7 = F.Linear(30, 1, initialW=np.zeros((1, 30), dtype=np.float32)),
                )
        """
        """
        self.critic_model = chainer.FunctionSet(
                l1 = F.Linear(2, 1),
                l2 = F.Linear(1, 1, initialW=np.zeros((1, 1), dtype=np.float32)),
                )
        self.actor_model = chainer.FunctionSet(
                l1 = F.Linear(2, 1),
                l2 = F.Linear(1, 1, initialW=np.zeros((1, 1), dtype=np.float32))
                )
        """
        if args.gpu >= 0:
            self.critic_model.to_gpu()
            self.actor_model.to_gpu()
        self.critic_optimizer = optimizers.SGD(self.ALPHA)
        self.critic_optimizer.setup(self.critic_model)
        self.actor_optimizer = optimizers.SGD(self.ALPHA)
        self.actor_optimizer.setup(self.actor_model)

        self.oldact = 0.0
        self.sigma = 10.0

        self.limit_action = 5.0
        self.min_action = -5.0
        self.max_action = 5.0
    
    def effort_callback(self, msg):
        pass
        #  print "effort : ", msg.data

    def get_joint_properties(self, joint_name):
        try:
            getjointproperties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            resp = getjointproperties(joint_name)
            #  print "resp.position : ", resp.position
            #  print "resp.rate : ", resp.rate
            return resp.position[0], resp.rate[0]
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e


    def reset_client(self, angle):
        q = tf.transformations.quaternion_from_euler(0.0, angle, 0.0)
        request = LinkState()
        request.link_name = "myrrbot6::myrrbot6_link2"
        request.pose.position.x = 0.0 + 2.0
        request.pose.position.y = 0.1 + 2.0
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
    
    def episode_end_flag(self, s):
        if math.fabs(s[0][0]) > 4*math.pi:
            return  True
        else:
            return False

    def reward_func(self, angle):
        r = math.cos(angle)
        #  print "r : ", r
        #  if r<0.0:
            #  r = 0.0
        return r

    def V_func(self, s):
        #  input_data = chainer.Variable(s)
        #  print "input_data.data : ", input_data.data
        #  h1 = F.relu(self.critic_model.l1(input_data))
        #  print "h1 : ", h1.data
        #  output_data = self.critic_model.l2(h1)
        input_data = chainer.Variable(s)
        h1 = F.relu(self.critic_model.l1(input_data))
        h2 = F.relu(self.critic_model.l2(h1))
        h3 = F.relu(self.critic_model.l3(h2))
        h4 = F.relu(self.critic_model.l4(h3))
        h5 = F.relu(self.critic_model.l5(h4))
        h6 = F.relu(self.critic_model.l6(h5))
        output_data = self.critic_model.l7(h6)
        return output_data

    def critic_forward(self, s, r, next_s, episode_end):
        V = self.V_func(s)
        #  print "V : ", V.data
        tmp = self.V_func(next_s)
        #  print "tmp : ", tmp.data

        if not episode_end:
            tmp_ = r + self.GAMMA * tmp.data[0][0]
        else:
            tmp_ = r

        target = xp.array([[tmp_]], dtype=xp.float32)

        td = chainer.Variable(target) - V

        zero_val = chainer.Variable(xp.zeros((1, 1), dtype=xp.float32))

        self.critic_optimizer.zero_grads()
        loss = F.mean_squared_error(td, zero_val)
        #  print "loss : ", loss.data

        loss.backward()
        #  print "self.critic_model.gradients : ", self.critic_model.gradients
        self.critic_optimizer.update()

        return loss, V, td

    def actor_func(self, s):
        input_data = chainer.Variable(s)
        h1 = F.relu(self.actor_model.l1(input_data))
        h2 = F.relu(self.actor_model.l2(h1))
        h3 = F.relu(self.actor_model.l3(h2))
        h4 = F.relu(self.actor_model.l4(h3))
        h5 = F.relu(self.actor_model.l5(h4))
        h6 = F.relu(self.actor_model.l6(h5))
        output_data = self.actor_model.l7(h6)
        return output_data

    def actor_forward(self, s):
        act = self.actor_func(s)
        #  print "act(actor) : ", act.data
        target = (act.data[0][0] + self.oldact) / 2.0
        #  print "target(actor) : ", target
        #  self.sigma = (self.sigma + (self.oldact - act.data[0][0])) / 2.0

        tmp = xp.array([[target]], dtype=xp.float32)
        

        error = chainer.Variable(tmp) - act
        #  print "error(actor) : ", error.data

        zero_val = chainer.Variable(xp.zeros((1, 1), dtype=xp.float32))
        
        self.actor_optimizer.zero_grads()
        
        loss = F.mean_squared_error(error, zero_val)
        #  print "loss(actor) : ", loss.data

        loss.backward()
        self.actor_optimizer.update()
        return loss, act

    def BoxMuller(self, mean, var):
        r1 = random()
        r2 = random()

        z1 = math.sqrt(-2.0 * math.log(r1))
        z2 = math.sin(2.0 * math.pi * r2)
        return var * z1 * z2 + mean

    def actor(self, s, train_flag=True):
        mu = self.actor_func(s)
        
        if train_flag:
            act = self.BoxMuller(mu.data[0][0], self.sigma)
            
            if act < self.min_action:
                while 1:
                    act += self.limit_action
                    if act > self.min_action:
                        break

            if act >= self.max_action:
                while 1:
                    act -= self.limit_action
                    if act <= self.max_action:
                        break
        else:
            act = mu.data[0][0]

        return act, mu


    def main(self, limit_episode, limit_step):
        rospy.init_node('deep_actor_critic_for_swingup_myrrbot6')

        rospy.Subscriber("/joint_effort", Float64, self.effort_callback)

        pub = rospy.Publisher("/myrrbot6/joint1_controller/command", Float64, queue_size = 1)

        loop_rate = rospy.Rate(20)

        count = 0
        wait_count = 0
        episode_count = 0
        time = 0.0

        V_list = np.array([])
        max_V = 0.0
        reward_list = np.array([])
        ave_reward = 0.0

        temp_result = np.array([[]])
        test_result = np.array([[]])

        #  init_angle = uniform(-1*self.pi, self.pi)
        init_angle = self.pi / 2.0

        while not rospy.is_shutdown():
            if(self.wait_flag):
                wait_count += 1
                if wait_count%20==0:
                    print "Please Wait %d seconds" % (1-wait_count/20)

                #  print "Reset!!!!"
                #  print "init_angle : ", init_angle
                self.reset_client(self.init_theta)
                self.init_omega = 0.0
                self.action = 0.0

                self.state[0][0] = self.init_theta
                self.state[0][1] = self.init_omega
                print "self.state : ", self.state
                pub.publish(self.action)
                
                if wait_count == 20*0.5:
                    self.init_theta = uniform(-1*self.pi, self.pi)
                if wait_count == 20*1:
                    wait_count = 0
                    self.wait_flag = False
            else:
                print ""
                #  print "self.critic_model.parameters : ", self.critic_model.parameters
                #  print "self.actor_model.parameters : ", self.actor_model.parameters
                print "Evaluation now!!!!!!!!!"
                print "Episode : %d / Time : %f" % (episode_count-1, time)
                count += 1
                time = float(count) / 20.0
                print "self.state(main) : ", self.state
                self.action, mu = self.actor(self.state, False)
                print "mu : ", mu.data
                print "self.action : %f" % (self.action)
                pub.publish(self.action)

                print "V.data : ", self.V_func(self.state).data
                
                self.next_state[0][0], self.next_state[0][1] = self.get_joint_properties('joint1')
                print "self.next_state(main) : ", self.next_state
                
                self.reward = self.reward_func(self.next_state[0][0])
                print "self.reward : ", self.reward
                
                if self.episode_end_flag(self.state):
                    print "Evaluation End!!!!!(rolling over 4*PI)"
                    count = 0
                    self.wait_flag = True
                    self.action = 0.0
                else:
                    self.state = self.next_state.copy()
                    self.oldact = self.action.copy()
                
                    if count >= 20*20:
                        print "Evaluation End!!!!!(over 20 seconds)"
                        count = 0
                        self.wait_flag = True
                        self.action = 0.0


            loop_rate.sleep()

if __name__=="__main__":
    deep_actor_critic_agent = agent()
    deep_actor_critic_agent.main(100000000, 100)

