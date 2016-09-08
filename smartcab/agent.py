import time
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import matplotlib.pyplot as plt
import math


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world.
       LearningAgent inherits from parents class Agent (defined in environment.py)
       So LearningAgent will get features(if not all) from its parent Agent class.
    """
        
    def __init__(self, env): # class constructor
#==============================================================================    
# super lets you call Agent class with parameter env, see class Agent from environment.py
# To learn more what super does and how you can use inheretance: 
# https://learnpythonthehardway.org/book/ex44.html
#==============================================================================     
        super(LearningAgent, self).__init__(env)  # sets self.env = env, self.state = None, self.next_waypoint = None, and a default color cyan
        self.color = 'red'  # override color
        self.planner = RoutePlanner(env, self)  # simple route planner to get next_waypoint
                                                     # no needs to redefine self.env, just env is OK 
        
        # TODO: Initialize any additional variables here
        self.state = {} # NoneType -> dict 
        self.valid_actions = [None, 'forward', 'left', 'right']
        self.q_values = {}  # Q dictionary of states/actions
        self.learning_rate = env.learning_rate
        self.discount_rate = env.discount_rate
        #self.learning_rate = 0.5
        #self.discount_rate = 0.05
        self.epsilon = 0.01
        self.penalty = 0
        self.total_reward = 0  
        self.err_time = 0
        
    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) #the stop light and traffic conditions the agent senses at the intersection it is currently at
        deadline = self.env.get_deadline(self)
    
        self.state = self.next_state(inputs)
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right']) #for random action
        #action = self.policy(self.state)
        
        action = self.epsilon_greedy(self.state, self.epsilon, t)
        
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.penalty += 1
            self.err_time = t
        self.total_reward += reward
        
        # TODO: Learn policy based on state, action, reward
        self.q_update(self.state, action, reward, self.learning_rate, self.discount_rate)
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print "LearningAgent.update_time(): time = {}".format(t)              
    
    def q_state(self, state, action):
        ''' States/actions : light(2), oncoming(4), left(4), direction(4) and action(4)'''
        return "light:{}, oncoming: {}, left: {}, direction: {}, action: {}".format(state['light'], state['oncoming'], state['left'], state['direction'], action)
    
    def q(self, state, action):
        '''Q value'''
        key = self.q_state(state, action)
        if key in self.q_values:
            return self.q_values[key]
        else:
            return 0
        
    def next_state(self, inputs):
        '''Next agent state with next_waypoint'''
        return {'light': inputs['light'], 'oncoming': inputs['oncoming'], 'left': inputs['left'], 'direction': self.next_waypoint}
        
    def q_update(self, state, action, reward, learning_rate, discount_rate):
        '''Update rule'''
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        key=self.q_state(state, action)
        q_current = self.q(state, action)
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.next_state(inputs)
    
        #Q(state, action) =  Q(state, action) + alpha(time) * (r + gamma * Q_max(next_state, all_action) - Q(state, action))
        q_new = q_current + self.learning_rate*(reward + self.discount_rate * self.q_max(new_state) - q_current)
        self.q_values[key] = q_new
        #print "Q_new: {} | key: {} | Q_max: {} q_current {}".format(q_new, key, self.q_max(new_state), q_current)
    
    def q_max(self,state):
        '''Choose q_max from all possible actions'''
        max = -1000
        for action in self.valid_actions:
            if (self.q(state, action)) > max:
                max = self.q(state, action)
        return max
    
    def policy(self, state):
        '''Value policy'''
        best_action = None
        q_best = 0
        for action in self.valid_actions:
            if self.q(state,action) > q_best:
                q_best = self.q(state, action)
                best_action = action
            if self.q(state, action) == q_best:
                best_action = random.choice([best_action, action])
        return best_action
    
    def epsilon_greedy(self, state, epsilon, t):
        ''' As time passes the agent do less and less exploration and follow policy best_actions'''
        if random.random() < epsilon/(math.exp(t)+15):
            return random.choice(self.valid_actions)
        else:
            return self.policy(state)
            
def run(learning_rate, discount_rate):
    '''Run the agent for a finite number of trials''' 
    goal = []
    penalties = []
    rewards = []
    penalty_rate = []
    N = 13    
    n = [pow(2,i) for i in xrange(N)]
    i = 0
    learning = []
    discount = []
    for n_trials in n:
        e = Environment(learning_rate, discount_rate)  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials   
        # Now simulate it
        sim = Simulator(e, update_delay=0, display=False)
        sim.run(n_trials)  # run for a specified number of trials
        goal.append(e.success)
        penalties.append(a.penalty)
        learning.append(a.learning_rate)
        discount.append(a.discount_rate)
        rewards.append(a.total_reward)
        print "Reached the goal in {} cases out of {} trials with {} penalties and reward sum {} at a time {} and last err time {}" \
        .format(goal[i], n_trials, penalties[i], rewards[i], e.t, a.err_time )
        penalty_rate.append(float(penalties[i])/n_trials)
        print "Penalty rate = {}".format(penalty_rate)
        i +=1
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
        df = pd.DataFrame({'Penalty_per_trial':pd.Series(penalty_rate), 
                           '#Trials':pd.Series(n),
                           'Successfull trials':pd.Series(goal), 
                           'Learning':pd.Series(learning), 
                           'Discount':pd.Series(discount),
                           })
    print df
    fig = plt.figure()
    fig.suptitle('learning_rate={}, discount_rate={}, epsilon={}'.format(a.learning_rate, a.discount_rate, a.epsilon), fontsize=10)
    plt.plot(df['Penalty_per_trial'], df['#Trials'], '-o')
    plt.yscale('log')       
    plt.ylabel('#Trials')
    plt.xlabel('Penalty per Trial')
    plt.axis([0, 10, -100, 6000])
    plt.show() #debug
    #df.to_csv('results.csv')
if __name__ == '__main__':
    start = time.clock()
    l_r = [0.5, 0.9]
    d_r = [0.05, 0.5]
    for i in l_r:
        for j in d_r:
            run(i,j)
    stop = time.clock()
    print "Time to calculate = {}".format(stop - start)
