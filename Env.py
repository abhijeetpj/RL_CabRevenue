# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        #self.action_space = list(permutations([i for i in range(m)], 2)) + [(0,0)] ## All permutaions of the actions and no action
        self.action_space = [(0,0),
                     (0,1), (0,2), (0,3), (0,4),
                     (1,0), (1,2), (1,3), (1,4),
                     (2,1), (2,0), (2,3), (2,4),
                     (3,1), (3,2), (3,0), (3,4),
                     (4,1), (4,2), (4,3), (4,0)]
        
        self.state_space = [(loc, tim, day) for loc in range(m) for tim in range(t) for day in range(d)] ##

        self.state_init = [np.random.randint(1,m), np.random.randint(1,t), np.random.randint(1,d)]
        
        self.days2terminal = 30            # to achieve terminal state OR an episode to complete
        self.maxhours2terminal = t * self.days2terminal
        self.total_ridetime = 0
        
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        # initiate a list of size (m+t+d) with 0
        encoded_state = [0] * (m + t + d)
        
        encoded_state[state[0]] = 1  ## set the location value into vector
        encoded_state[m+state[1]] = 1
        encoded_state[m+t+state[2]] = 1## set time value into vector
        return encoded_state

    def action_encod_arch1(self, action):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        encoded_action = [0] * (m + m)
        if action[0] != 0 and action[1] != 0:
            encoded_action[action[0]] = 1     ## set pickup location:
            encoded_action[m + action[1]] = 1     ## set drop location
        return encoded_action


    # Use this function if you are using architecture-2
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        encoded_state = [0] * (m+t+d+m+m)  ## initialize vector state + action space
        encoded_state[state[0]] = 1  ## set the location value into vector
        encoded_state[m+state[1]] = 1  ## set time value into vector
        encoded_state[m+t+state[2]] = 1  ## set day value into vector
        if (action[0] != 0):
            encoded_state[m+t+d+action[0]] = 1     ## set pickup location
        if (action[1] != 0):
            encoded_state[m+t+d+m+action[1]] = 1     ## set drop location
        return encoded_state


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]


        actions.append((0,0))

        # Update index for [0, 0]

        possible_actions_index.append(self.action_space.index((0,0)))

        return possible_actions_index, actions


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        if (pickup_loc) == 0 and (drop_loc == 0):
            reward = -C

        ## 2. Driver wants to have pickup and is at same location
        elif pickup_loc == curr_loc:
            rideTime = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            
            reward = (R * rideTime) - (C * rideTime)

        ## 3. Driver wants to pickup and is at different location
        else:
            pickup_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            updated_time, updated_day = self.calc_updated_day_time(curr_time, curr_day, pickup_time)
            
            rideTime =  Time_matrix[pickup_loc][drop_loc][updated_time][updated_day]
            
            reward = (R * rideTime) - (C * pickup_time) - (C * rideTime)

        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        ## Find current state of driver
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]

        ## reward depends of time, lets initialize
        total_time = 0
        norideTime = 0
        ride_time = 0
        pickup_time = 0

        ## next state depends on possible actions taken by cab driver
        ## There are 3 cases for the same
        ## 1. Cab driver refuse i.e. action (0,0)
        if (pickup_loc) == 0 and (drop_loc == 0):
            norideTime = 1
            pickup_time = 0
            ride_time = 0
            
            next_loc = curr_loc
            curr_time = curr_time + norideTime
            
            updated_time, updated_day = self.calc_updated_day_time(curr_time, curr_day, norideTime)
            curr_time = updated_time
            curr_day = updated_day

        ## 2. Driver wants to have pickup and is at same location
        elif pickup_loc == curr_loc:
            pickup_time = 0
            norideTime = 0
            
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            updated_time, updated_day = self.calc_updated_day_time(curr_time, curr_day, ride_time)
            
            next_loc = drop_loc
            curr_time = updated_time
            curr_day = updated_day

        ## 3. Driver wants to pickup and is at different location
        else:
            norideTime = 0
            pickup_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            updated_time, updated_day = self.calc_updated_day_time(curr_time, curr_day, pickup_time)
            
            ride_time =  Time_matrix[pickup_loc][drop_loc][updated_time][updated_day]
            updated_time, updated_day = self.calc_updated_day_time(updated_time, updated_day, ride_time)
            
            next_loc  = drop_loc
            curr_time = updated_time
            curr_day = updated_day

        total_time = norideTime + pickup_time + ride_time
        #print('total travel time',total_time)
        #updated_time, updated_day = self.calc_updated_day_time(curr_time, curr_day, total_time)
        #print('updated time',updated_time)
        #print('updated day',updated_day)
        next_state = [next_loc, curr_time, curr_day]

        return next_state, total_time

    def calc_updated_day_time(self, time, day, duration):
        '''
        Takes the current day, time and time duration and returns updated day and time based on the time duration of travel
        '''
        if time >= t:
            time = time - t
            day = day + 1
            if day >= d:
                day = day - d
                
        return time, day

    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init


c = CabDriver()
print(c.requests([0, 1, 5]))
