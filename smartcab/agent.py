import time
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

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
        self.learning_rate = 0.5
        self.discount_rate = 0.05
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) #the stop light and traffic conditions the agent senses at the intersection it is currently at
        deadline = self.env.get_deadline(self)
    
        # TODO: Update state
        self.state = self.next_state(inputs)
        # TODO: Select action according to your policy
        #action = random.choice([None, 'forward', 'left', 'right']) #for random action
        action = self.policy(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.q_update(self.state, action, reward)
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    
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
        
    def q_update(self, state, action, reward):
        '''Update rule'''
        key=self.q_state(state, action)
        q_current = self.q(state, action)
        inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = self.next_state(inputs)
    
        #Q(state, action) =  Q(state, action) + alpha(time) * (r + gamma * Q_max(next_state, all_action) - Q(state, action))
        q_new = q_current + self.learning_rate*(reward + self.discount_rate * self.q_max(new_state) - q_current)
        self.q_values[key] = q_new
        print "Q_new: {} | key: {} | Q_max: {} q_current {}".format(q_new, key, self.q_max(new_state), q_current)
          
    def q_max(self,state):
        '''Choose q_max from all possible actions'''
        max = -1000
        for action in self.valid_actions:
            if (self.q(state, action)) > max:
                max = self.q(state, action)
        return max
    
    def policy(self, state):
        '''Simple policy which allow to increase Q based on (state, action)'''

        best_action = None
        q_best = 0 
        for action in self.valid_actions:
            
            if self.q(state, action) > q_best:
                q_best = self.q(state, action)
                best_action = action
            if self.q(state, action) == q_best:
                best_action = random.choice([best_action, action])
        return best_action
          
def run():
    start = time.clock()
    '''Run the agent for a finite number of trials'''    
    n_values = [1, 10, 100, 1000]
    goal = []
    penalties = []  
    for n_trials in n_values:
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials   
        # Now simulate it
        sim = Simulator(e, update_delay=0, display=False)
        sim.run(n_trials)  # run for a specified number of trials
        goal.append(e.sucess)
        penalties.append(e.penalty)
        print "Reached the goal in {} cases out of {} trials with {} penalties".format(goal, n_values, penalties)
    stop = time.clock()
    print stop - start
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
