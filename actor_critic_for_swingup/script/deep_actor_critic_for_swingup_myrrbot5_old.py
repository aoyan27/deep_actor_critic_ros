#!/usr/bin/env python
#coding:utf-8
import argparse

import rospy

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

from std_msgs.msg import Float64, Int64, Float64MultiArray, String
from gazebo_msgs.msg import*
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from sensor_msgs.msg import*

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
    #  ALPHA = 0.001
    GAMMA = 0.9

    wait_flag = True
    
    filename_critic_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/models/myrrbot5_critic_model.dat"
    filename_actor_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/models/myrrbot5_actor_model.dat"

    filename_result = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/results/myrrbot5_result.txt"

    def __init__(self):
        self.init_theta = 0.0
        self.init_omega = 0.0
        
        self.episode_time = 10.0
        self.hz = 20.0

        self.evaluation_freq = 100

        self.state = xp.zeros((1, 5), dtype=xp.float32)
        self.next_state = xp.zeros((1, 5), dtype=xp.float32)
        self.reward = 0.0
        self.action = 0.0

        self.critic_model = chainer.FunctionSet(
                l1 = F.Linear(5, 500),
                l2 = F.Linear(500, 250),
                l3 = F.Linear(250, 125),
                l4 = F.Linear(125, 60),
                l5 = F.Linear(60, 30),
                l6 = F.Linear(30, 15),
                l7 = F.Linear(15, 1, initialW=np.zeros((1, 15), dtype=np.float32)),
                )
        self.actor_model = chainer.FunctionSet(
                l1 = F.Linear(5, 500),
                l2 = F.Linear(500, 250),
                l3 = F.Linear(250, 125),
                l4 = F.Linear(125, 60),
                l5 = F.Linear(60, 30),
                l6 = F.Linear(30, 15),
                l7 = F.Linear(15, 1, initialW=np.zeros((1, 15), dtype=np.float32)),
                )
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
    
    def get_link_state(self, link_name, reference_frame_name):
        try:
            getlinkstate = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            resp = getlinkstate(link_name, reference_frame_name)
            #  print "resp.position : ", resp.position
            #  print "resp.rate : ", resp.rate
            quaternion = (
                    resp.link_state.pose.orientation.x,
                    resp.link_state.pose.orientation.y,
                    resp.link_state.pose.orientation.z,
                    resp.link_state.pose.orientation.w)
            return quaternion, resp.link_state.twist.angular
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e


    def get_joint_properties(self, joint_name):
        try:
            getjointproperties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            resp = getjointproperties(joint_name)
            #  print "resp.position : ", resp.position
            #  print "resp.rate : ", resp.rate
            return resp.position[0]
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e


    def reset_client(self, angle):
        #  print "angle : ", angle
        q = tf.transformations.quaternion_from_euler(0.0, angle, 0.0)
        request = LinkState()
        request.link_name = "myrrbot5::myrrbot5_link2"
        request.pose.position.x = 1.0
        request.pose.position.y = -0.9
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
    
    def episode_end_flag(self, theta):
        print "theta : ", theta
        if math.fabs(theta) > 3*math.pi:
            return True
        else:
            return False

    def reward_func(self, angle):
        r = math.cos(angle)
        #  print "r : ", r
        #  if r >= 0.0:
            #  r = 1.0
        #  else:
            #  r = -1.0
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

        loss.backward()
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

        zero_val = chainer.Variable(xp.zeros((1, 1), dtype=xp.float32))
        self.actor_optimizer.zero_grads()
        loss = F.mean_squared_error(error, zero_val)

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
        act = self.actor_func(s)
        
        if train_flag:
            action = act.data[0][0] + self.BoxMuller(0.0, 5.0)
            #  print "training(action) : ", action
            
            if action < self.min_action:
                while 1:
                    #  print "training(action) : ", action
                    action += self.limit_action
                    if action > self.min_action:
                        break

            if action >= self.max_action:
                while 1:
                    #  print "tarining(action)_ : ", action
                    action -= self.limit_action
                    if action <= self.max_action:
                        break
        else:
            action = act.data[0][0]

        #  print "action(actor) : ", action
        #  print "act(actor) : ", act.data

        return action, act

    def save_result(self, result):
        f_critic_model = open(self.filename_critic_model, 'w')
        f_actor_model = open(self.filename_actor_model, 'w')
        f_result = open(self.filename_result, 'w')
        evaluation_flag =True
        np.savetxt(self.filename_result, result, fmt="%.6f", delimiter=",")
        pickle.dump(self.critic_model, f_critic_model)
        pickle.dump(self.actor_model, f_actor_model)

        f_critic_model.close()
        f_actor_model.close()
    
    def main(self, limit_episode, limit_step):
        rospy.init_node('deep_actor_critic_for_swingup_myrrbot5')

        rospy.Subscriber("/joint_effort", Float64, self.effort_callback)

        pub = rospy.Publisher("/myrrbot5/joint1_controller/command", Float64, queue_size = 1)

        loop_rate = rospy.Rate(self.hz)

        count = 0
        wait_count = 0
        episode_count = 0
        time = 0.0

        V_list = np.array([])
        max_V = 0.0
        ave_V = 0.0
        reward_list = np.array([])
        ave_reward = 0.0
        total_reward = 0.0

        temp_result = np.array([[]])
        test_result = np.array([[]])

        #  init_angle = uniform(-1*self.pi, self.pi)
        init_angle = self.pi / 2.0

        evaluation_flag = False


        while not rospy.is_shutdown():
            if(self.wait_flag):
                wait_count += 1
                #  if wait_count%self.hz==0:
                    #  print "Please Wait %d seconds" % (1-wait_count/self.hz)

                #  print "Reset!!!!"
                #  print "init_angle : ", init_angle
                self.reset_client(self.init_theta)
                self.init_omega = 0.0
                self.action = 0.0

                pub.publish(self.action)
                
                if wait_count == self.hz*0.5:
                    self.init_theta = uniform(-1*self.pi, 1*self.pi)
                    #  self.init_theta = uniform(0.0, 2.0*self.pi)
                    q = tf.transformations.quaternion_from_euler(0.0, self.init_theta, 0.0)
                    self.state[0][0] = q[0]
                    self.state[0][1] = q[1]
                    self.state[0][2] = q[2]
                    self.state[0][3] = q[3]
                    self.state[0][4] = self.init_omega
                    print "self.state : ", self.state
                if wait_count == self.hz*1:
                    wait_count = 0
                    self.wait_flag = False
            else:
                print ""
                #  print "self.critic_model.parameters : ", self.critic_model.parameters
                #  print "self.actor_model.parameters : ", self.actor_model.parameters
                if not evaluation_flag:
                    
                    #  if time == 0:
                        #  if episode_count%10 == 0:
                            #  print "Episode : %d" % (episode_count)
                    print "Episode : %d / Time : %f" % (episode_count, time)
                    #  now = rospy.get_rostime()
                    #  rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
                    count += 1
                    time = float(count) / self.hz
                    print "self.state : ", self.state
                    self.action, act = self.actor(self.state)
                    print "mu : ", act.data[0][0]
                    print "self.action : %f" % (self.action)
                    pub.publish(self.action)

                    quaternion, angular = self.get_link_state('myrrbot5::myrrbot5_link2', '')
                    print "quaternion : ", quaternion
                    print "angular : ", angular
                    self.next_state[0][0] = quaternion[0] 
                    self.next_state[0][1] = quaternion[1]
                    self.next_state[0][2] = quaternion[2]
                    self.next_state[0][3] = quaternion[3]
                    self.next_state[0][4] = angular.y
                    print "self.next_state : ", self.next_state


                    #  qu = Quaternion(0.0, 0.0, 0.7071, 0.7071)
                    #  print "0.0 %f 0.0" % (-60.0/180.0*math.pi)
                    #  qu = tf.transformations.quaternion_from_euler(0.0, 1.9, 0.0)
                    #  print "qu : ", qu
                    #  euler = tf.transformations.euler_from_quaternion((qu[0], qu[1], qu[2], qu[3]))
                    #  print "euler : ", euler
                    #  euler = tf.transformations.euler_from_quaternion(quaternion)
                    #  qu1 = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
                    #  print "qu1 : ", qu1
                    
                    self.reward = self.reward_func(self.get_joint_properties('myrrbot5_joint1'))
                    print "self.reward : ", self.reward
                    #  reward_list = np.append(reward_list, [self.reward])

                    #  print "self.state = %s, self.next_state=%s" % (self.state, self.next_state)
                    loss_critic, V, td = self.critic_forward(self.state, self.reward, self.next_state, False)
                    #  print "loss_critic : ", loss_critic.data
                    print "V : ", V.data
                    #  V_list = np.append(V_list, [float(V.data[0][0])])
                    print "td : ", td.data
                    #  loss_actor, act = self.actor_forward(self.state)
                    
                    if td.data[0][0] > 0:
                        loss_actor, act = self.actor_forward(self.state)
                        

                    if self.episode_end_flag(self.get_joint_properties('myrrbot5_joint1')):
                        print "Episode End!!!!(rolling over 4*PI)"
                        #  print "V_list : ", V_list
                        #  max_V = np.max(V_list)
                        #  print "max_V : ", max_V
                        #  ave_V = np.average(V_list)
                        #  print "ave_V : ", ave_V
                        #  V_list = np.array([])
                        #  print "reward_lsit : ", reward_list
                        #  ave_reward = np.average(reward_list)
                        #  print "ave_reward : ", ave_reward
                        #  reward_list = np.array([])

                        #  temp_result = np.array(([[episode_count, max_V, ave_V, ave_reward]]), dtype=np.float32)
                        #  print "temp_result : ", temp_result
                        #  if episode_count == 0:
                            #  print "test_result : ", test_result
                            #  test_result = temp_result
                            #  print "test_result : ", test_result
                        #  else:
                            #  test_result = np.r_[test_result, temp_result]
                        #  print "test_result : ", test_result
                        
                        if episode_count%self.evaluation_freq==0:
                            #  self.save_result(test_result)
                            evaluation_flag =True
                        
                        count = 0
                        episode_count += 1
                        self.wait_flag = True
                        self.action = 0.0
                    
                    else:
                        self.state = self.next_state.copy()
                        self.oldact = self.action.copy()
                    
                        if count >= self.episode_time*self.hz:
                            print "Episode End!!!!(over %d seconds)" % self.episode_time
                            #  print "V_list : ", V_list
                            #  max_V = np.max(V_list)
                            #  print "max_V : ", max_V
                            #  ave_V = np.average(V_list)
                            #  print "ave_V : ", ave_V
                            #  V_list = np.array([])
                            #  print "reward_lsit : ", reward_list
                            #  ave_reward = np.average(reward_list)
                            #  print "ave_reward : ", ave_reward
                            #  reward_list = np.array([])
                            
                            #  temp_result = np.array(([[episode_count, max_V, ave_V, ave_reward]]), dtype=np.float32)
                            #  print "temp_result : ", temp_result
                            
                            #  if episode_count == 0:
                                #  print "test_result : ", test_result
                                #  test_result = temp_result
                                #  print "test_result : ", test_result
                            #  else:
                                #  test_result = np.r_[test_result, temp_result]
                            #  print "test_result : ", test_result
                            if episode_count%self.evaluation_freq==0:
                                #  self.save_result(test_result)
                                evaluation_flag =True
                            
                            count = 0
                            episode_count += 1
                            self.wait_flag = True
                            self.action = 0.0

                    if episode_count >= limit_episode:
                        print "Finish!!!!!"
                        break
                else:
                    #  print ""
                    print ""
                    print "Evaluation now!!!!!!!!!"
                    print "Episode : %d / Time : %f" % (episode_count-1, time)
                    #  now = rospy.get_rostime()
                    #  rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
                    count += 1
                    time = float(count) / self.hz
                    print "self.state(main) : ", self.state
                    self.action, mu = self.actor(self.state, False)
                    print "mu : ", mu.data
                    print "self.action : %f" % (self.action)
                    pub.publish(self.action)
                    
                    quaternion, angular = self.get_link_state('myrrbot5::myrrbot5_link2', '')
                    print "quaternion : ", quaternion
                    print "angular : ", angular
                    self.next_state[0][0] = quaternion[0] 
                    self.next_state[0][1] = quaternion[1]
                    self.next_state[0][2] = quaternion[2]
                    self.next_state[0][3] = quaternion[3]
                    self.next_state[0][4] = angular.y
                    print "self.next_state : ", self.next_state

                    
                    self.reward = self.reward_func(self.get_joint_properties('myrrbot5_joint1'))
                    print "self.reward : ", self.reward
                    reward_list = np.append(reward_list, [self.reward])
                    
                    V = self.V_func(self.state)
                    print "V : ", V.data
                    V_list = np.append(V_list, [float(V.data[0][0])])
                    
                    if self.episode_end_flag(self.get_joint_properties('myrrbot5_joint1')):
                        print "Evaluation End!!!!!(rolling over 4*PI)"
                        max_V = np.max(V_list)
                        print "max_V : ", max_V
                        ave_V = np.average(V_list)
                        print "ave_V : ", ave_V
                        V_list = np.array([])
                        #  print "reward_lsit : ", reward_list
                        ave_reward = np.average(reward_list)
                        print "ave_reward : ", ave_reward
                        total_reward = np.sum(reward_list)
                        print "total_reward : ", total_reward
                        reward_list = np.array([])

                        temp_result = np.array(([[episode_count-1.0, max_V, ave_V, ave_reward, total_reward]]), dtype=np.float32)
                        #  print "temp_result : ", temp_result
                        if episode_count-1 == 0:
                            #  print "test_result : ", test_result
                            test_result = temp_result
                            #  print "test_result : ", test_result
                        else:
                            test_result = np.r_[test_result, temp_result]
                        #  print "test_result : ", test_result
                        self.save_result(test_result)
                        
                        count = 0
                        self.wait_flag = True
                        self.action = 0.0
                        evaluation_flag = False
                    else:
                        self.state = self.next_state.copy()
                        self.oldact = self.action.copy()
                    
                        if count >= self.episode_time*self.hz:
                            print "Evaluation End!!!!!(over %d seconds)" % self.episode_time
                            max_V = np.max(V_list)
                            print "max_V : ", max_V
                            ave_V = np.average(V_list)
                            print "ave_V : ", ave_V
                            V_list = np.array([])
                            #  print "reward_lsit : ", reward_list
                            ave_reward = np.average(reward_list)
                            print "ave_reward : ", ave_reward
                            total_reward = np.sum(reward_list)
                            print "total_reward : ", total_reward
                            reward_list = np.array([])

                            temp_result = np.array(([[episode_count-1.0, max_V, ave_V, ave_reward, total_reward]]), dtype=np.float32)
                            #  print "temp_result : ", temp_result
                            if episode_count-1 == 0:
                                #  print "test_result : ", test_result
                                test_result = temp_result
                                #  print "test_result : ", test_result
                            else:
                                test_result = np.r_[test_result, temp_result]
                            #  print "test_result : ", test_result
                            self.save_result(test_result)
                            
                            count = 0
                            self.wait_flag = True
                            self.action = 0.0
                            evaluation_flag = False


            loop_rate.sleep()

if __name__=="__main__":
    deep_actor_critic_agent = agent()
    deep_actor_critic_agent.main(100000000, 100)

