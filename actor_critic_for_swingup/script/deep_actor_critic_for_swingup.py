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
    
    #  ALPHA = 1e-6
    ALPHA = 0.001
    GAMMA = 0.9

    wait_flag = True

    def __init__(self):
        self.state = xp.array([[0.0, 0.0]], dtype=np.float32)
        self.next_state = xp.array([[0.0, 0.0]], dtype=np.float32)
        self.reward = 0.0
        self.action = 0.0

        self.critic_model = chainer.FunctionSet(
                l1 = F.Linear(2, 100),
                l2 = F.Linear(100, 1, initialW=np.zeros((1, 100), dtype=np.float32)),
                )
        self.actor_model = chainer.FunctionSet(
                l1 = F.Linear(2, 100),
                l2 = F.Linear(100, 1, initialW=np.zeros((1, 100), dtype=np.float32)),
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

    def state_callback(self, msg):
        print "Subscribe state!!!!!!!!!!"
        self.state_data = xp.array([[msg.data[0], msg.data[1]]], dtype=xp.float32)
        print "self.state_data : ", self.state_data
        #  self.state_data = s
        #  print "msg.data[0] : ", msg.data[0]
        #  print "msg.data[1] : ", msg.data[1]
        #  print "type(msg.data[0]) : ", type(msg.data[0])
        #  print "type(msg.data[1]) : ", type(msg.data[1])
        #  print "self.state_data : ", self.state_data
    
    def effort_callback(self, msg):
        pass
        #  print "effort : ", msg.data

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
    
    def episode_end_flag(self, s):
        return False

    def reward_func(self, angle):
        r = math.cos(angle)
        #  print "r : ", r
        if r<0.0:
            r = 0.0
        return r

    def V_func(self, s):
        input_data = chainer.Variable(s)
        #  print "input_data.data : ", input_data.data
        h1 = F.relu(self.critic_model.l1(input_data))
        #  print "h1 : ", h1.data
        output_data = self.critic_model.l2(h1)
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
        output_data = self.actor_model.l2(h1)
        return output_data

    def actor_forward(self, s):
        act = self.actor_func(s)
        #  print "act(actor) : ", act.data
        target = (act.data[0][0] + self.oldact) / 2.0
        #  print "target(actor) : ", target
        #  self.sigma = (self.sigma + (self.oldact - act.data[0][0])) / 2.0
        if self.sigma > 0.01:
            self.sigma -= 1e-3
        else:
            self.sigma = 0.01

        tmp = xp.array([[target]], dtype=xp.float32)
        

        error = chainer.Variable(tmp) - act

        zero_val = chainer.Variable(xp.zeros((1, 1), dtype=xp.float32))
        loss = F.mean_squared_error(error, zero_val)

        loss.backward()
        self.actor_optimizer.zero_grads()
        self.actor_optimizer.update
        return loss, act

    def BoxMuller(self, mean, var):
        r1 = random()
        r2 = random()

        z1 = math.sqrt(-2.0 * math.log(r1))
        z2 = math.sin(2.0 * math.pi * r2)
        return var * z1 * z2 + mean

    def actor(self, s):
        mu = self.actor_func(s)

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
        self.oldact = act

        return act



    def main(self, limit_episode, limit_step):
        rospy.init_node('deep_actor_critic_for_swingup')

        rospy.Subscriber("/state", Float64MultiArray, self.state_callback)
        rospy.Subscriber("/joint_effort", Float64, self.effort_callback)

        pub = rospy.Publisher("/action", Float64, queue_size = 1)

        loop_rate = rospy.Rate(50)

        count = 0
        wait_count = 0
        episode_count = 1
        time = 0.0

        #  init_angle = uniform(-1*self.pi, self.pi)
        init_angle = self.pi / 2.0

        while not rospy.is_shutdown():
            if(self.wait_flag):
                wait_count += 1
                if wait_count%50==0:
                    print "Please Wait %d seconds" % (5-wait_count/50)

                #  print "Reset!!!!"
                #  print "init_angle : ", init_angle
                self.reset_client(init_angle)
                self.action = 0.0
                pub.publish(self.action)
                if wait_count == 50*1:
                    wait_count = 0
                    init_angle = uniform(-1*self.pi, self.pi)

                    self.wait_flag = False

            else:
                print ""
                #  print "self.critic_model.parameters : ", self.critic_model.parameters
                #  print "self.actor_model.parameters : ", self.actor_model.parameters
                print "Episode : %d / Time : %f" % (episode_count, time)
                count += 1
                time = float(count) / 50.0
                print "self.state(main) : ", self.state
                self.action = self.actor(self.state)
                print "self.action(self.sigma) : %f(%f)" % (self.action, self.sigma)
                pub.publish(self.action)

                self.next_state = self.state_data
                print "self.next_state(main) : ", self.next_state
                
                self.reward = self.reward_func(self.next_state[0][0])
                print "self.reward : ", self.reward
                
                loss_critic, V, td = self.critic_forward(self.state, self.reward, self.next_state, False)
                #  print "loss_critic : ", loss_critic.data
                print "V : ", V.data
                print "td : ", td.data
                
                if td.data[0][0] > 0:
                    loss_actor, act = self.actor_forward(self.state)
                    #  print "loss_actor : ", loss_actor.data
                    print "act : ", act.data
                    print "self.sigma : ", self.sigma
                else:
                    if self.sigma < 10.0:
                        self.sigma += 1e-3
                    else:
                        self.sigma = 10.0

                if self.episode_end_flag(self.state):
                    count = 0
                    episode_count += 1
                    self.wait_flag = True
                    self.action = 0.0
                else:
                    self.state = self.next_state
                
                if count >= 20*50:
                    count = 0
                    episode_count += 1
                    self.wait_flag = True
                    self.action = 0.0

                if episode_count >= limit_episode:
                    print "Finish!!!!!"
                    break


            loop_rate.sleep()

if __name__=="__main__":
    deep_actor_critic_agent = agent()
    deep_actor_critic_agent.main(10000000, 100)

