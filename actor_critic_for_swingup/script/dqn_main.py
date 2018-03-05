#!/usr/bin/env python
#coding:utf-8

import rospy

import gym, sys
import numpy as np

from std_msgs.msg import Float64, Int64, Float64MultiArray, String
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

import tf

import math
from random import random, randint, uniform

from dqn_agent import Agent

max_torque = 5.0
min_torque = -5.0
hz = 20


def get_joint_properties(joint_name):
    try:
        getjointproperties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
        resp = getjointproperties(joint_name)
        #  print "resp.position : ", resp.position
        #  print "resp.rate : ", resp.rate
        return resp.position[0], resp.rate[0]
    except rospy.ServiceException, e:
        print "Service call faild : %s" % e

def reset_client(angle):
    q = tf.transformations.quaternion_from_euler(0.0, angle, 0.0)
    request = LinkState()
    request.link_name = "myrrbot7::myrrbot7_link2"
    request.pose.position.x = 0.0 + 2.0
    request.pose.position.y = 0.1 - 2.0
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


def get_state(theta, omega):
    return np.array([[np.cos(theta), np.sin(theta), omega]], dtype=np.float32)

def get_reward(theta, omega, action):
    return np.cos(theta)
    #  return angle_normalizer(theta)**2 + 0.1 * omega**2 + 0.001*(action**2)

def get_ep_end(theta, time_step, step_limit):
    if theta > 3 * math.pi:
        return False
    else:
        if time_step < step_limit-1:
            return False
        else:
            return True

def save_result(path_name, result):
    f_result = open(path_name, 'w')
    np.savetxt(path_name, result, fmt="%.6f", delimiter=",")

def angle_normalizer(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def main(limit_episode, limit_step, seed=0, load=False):
    rospy.init_node('dqn_pendulum_myrrbot7')

    pub = rospy.Publisher("/myrrbot7/joint1_cotroller/command", Float64, queue_size=1)

    loop_rate = rospy.Rate(hz)
    
    result_path = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/results/dqn_result.txt"
    model_path = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/models/dqn_"

    init_theta = 0.0
    init_omega = 0.0

    state = get_state(init_theta, init_omega)
    state_dash = get_state(init_theta, init_omega)
    action = 0.0
    reward = 0.0
    action_list = [np.array([a], dtype=np.float64) for a in [min_torque, max_torque]]
    n_st = len(state[0])
    n_act = len(action_list)

    Q_list = np.array([])
    max_Q = 0.0
    ave_Q = 0.0
    reward_list = np.array([])
    ave_reward = 0.0
    total_reward = 0.0

    temp_result = np.array([[]])
    test_result = np.array([[]])

    evaluation_flag = False
    wait_flag = True
    
    agent = Agent(n_st, n_act, seed)

    if load:
        agent.load_model(model_path)
        
    episode_count = 0
    time = 1
    count = 0
    wait_count = 0


    while not rospy.is_shutdown():
        if wait_flag:
            wait_count += 1
            #  print "wait_count : ", wait_count

            state = get_state(init_theta, init_omega)
            state_dash = get_state(init_theta, init_omega)
            reset_client(init_theta)
            action = 0.0
            pub.publish(action)

            if wait_count == 0.5 * hz:
                init_theta = uniform(-1*math.pi, math.pi)

            if wait_count % hz == 0:
                wait_count = 0
                wait_flag = False
                #  print "Please Wait 1 second"
                #  print "state : ", state
                #  print "state_dash : ", state_dash
        else:
            if not evaluation_flag:
                #  print "Now Learning!!!!!"
                #  print "episode : ", episode_count
                #  print "time : ", time
                #  print "state : ", state
                act_i, q = agent.get_action(state, False)
                Q_list = np.append(Q_list, [q])
                #  print "act_i : ", act_i
                action = action_list[act_i]
                #  print "action : ", action
                pub.publish(action)
                theta, omega = get_joint_properties('myrrbot7_joint1')
                #  print "theta : %f, omega : %f" % (theta, omega)
                state_dash = get_state(theta, omega)
                #  print "state_dash : ", state_dash
                reward = get_reward(theta, omega, action)
                reward_list = np.append(reward_list, [reward])
                #  print "reward : ", reward
                ep_end = get_ep_end(theta, time, limit_step)
                #  print "ep_end : ", ep_end
                agent.stock_experience(count, state, act_i, reward, state_dash, ep_end)
                agent.train(count)
                
                time += 1
                count += 1

                if ep_end:
                    max_Q = np.max(Q_list)
                    ave_Q = np.average(Q_list)
                    #  print "max_Q : ", max_Q
                    #  print "ave_Q : ",ave_Q
                    ave_reward = np.average(reward_list)
                    total_reward = np.sum(reward_list)
                    #  print "ave_reward : ", ave_reward
                    #  print "total_reward : ", total_reward

                    print "Episode : %d\t/Reward Sum : %f\tEpsilon : %f\tLoss : %f\t/Average Q : %f\t/Time Step : %d" % (episode_count, total_reward, agent.epsilon, agent.loss, np.sum(Q_list)/float(time), agent.step+1)
                    Q_list = np.array([])
                    reward_list = np.array([])
                    temp_result = np.array(([[episode_count, max_Q, ave_Q, ave_reward, total_reward]]), dtype=np.float32)

                    if episode_count == 0:
                        #  print "test_result : ", test_result
                        test_result = temp_result
                        #  print "test_result : ", test_result
                    else:
                        test_result = np.r_[test_result, temp_result]

                    save_result(result_path, test_result)
                    agent.save_model(model_path)
            
                    if episode_count % 1 == 0:
                        evaluation_flag = False
                    
                    episode_count += 1
                    time = 0
                    wait_flag = True
            else:
                #  print "Now evaluation!!!"
                #  print "episode : ", episode_count-1
                #  print "time : ", time
                #  print "state : ", state
                act_i, q = agent.get_action(state, True)
                Q_list = np.append(Q_list, [q])
                #  print "act_i : ", act_i
                #  print "Q_list : ", Q_list
                action = action_list[act_i]
                #  print "action : ", action
                pub.publish(action)
                theta, omega = get_joint_properties('myrrbot7_joint1')
                #  print "theta : %f, omega : %f" % (theta, omega)
                state_dash = get_state(theta, omega)
                #  print "state_dash : ", state_dash
                reward = get_reward(theta, omega, action)
                #  print "reward : ", reward
                reward_list = np.append(reward_list, [reward])
                #  print "reward_list : ", reward_list
                ep_end = get_ep_end(theta, time, limit_step)
                #  print "ep_end : ", ep_end
                if ep_end:
                    max_Q = np.max(Q_list)
                    ave_Q = np.average(Q_list)
                    #  print "max_Q : ", max_Q
                    #  print "ave_Q : ",ave_Q
                    ave_reward = np.average(reward_list)
                    total_reward = np.sum(reward_list)
                    #  print "ave_reward : ", ave_reward
                    #  print "total_reward : ", total_reward

                    print "Episode : %d\t/Reward Sum : %f\tEpsilon : %f\tLoss : %f\t/Average Q : %f\t/Time Step : %d" % (episode_count-1, total_reward, agent.epsilon, agent.loss, np.sum(Q_list)/float(time+1), agent.step)
                    Q_list = np.array([])
                    reward_list = np.array([])
                    
                    time = 0
                    wait_flag = True
                    evaluation_flag = False

                    temp_result = np.array(([[episode_count-1, max_Q, ave_Q, ave_reward, total_reward]]), dtype=np.float32)

                    if episode_count-1 == 0:
                        #  print "test_result : ", test_result
                        test_result = temp_result
                        #  print "test_result : ", test_result
                    else:
                        test_result = np.r_[test_result, temp_result]

                    save_result(result_path, test_result)
                    agent.save_model(model_path)
                    
                
                time += 1



        loop_rate.sleep()

if __name__=="__main__":
    #  env_name = sys.argv[1]
    main(1000, 200)


