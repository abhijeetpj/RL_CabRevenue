# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
locations = 5 # number of cities, ranges from 1 ..... m
max_hours = 24 # number of hours, ranges from 0 .... t-1
max_days = 7  # number of days, ranges from 0 ... d-1
cost = 5 # Per hour fuel and other costs
revenue = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # action_space will of (p,q) where p->pickup location, q->drop location
        self.action_space = [(0,0),
                             (0,1), (0,2), (0,3), (0,4),
                             (1,0), (1,2), (1,3), (1,4),
                             (2,1), (2,0), (2,3), (2,4),
                             (3,1), (3,2), (3,0), (3,4),
                             (4,1), (4,2), (4,3), (4,0)]
        
        #state_space id defined by current location, hour of the day and day of the week i.e (locations, max_hours, max_days)
        self.state_space = [[loc,hour,day] for loc in range(locations) for hour in range(max_hours) for day in range(max_days)]
        
        self.state_init = [np.random.randint(1,locations), np.random.randint(1,max_hours), np.random.randint(1,max_days)]

        self.days2terminal = 30            # to achieve terminal state OR an episode to complete
        self.maxhours2terminal = max_hours * self.days2terminal
        self.total_ridetime = 0
        
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        # initiate a list of size (m+t+d) with 0
        state_encoded = [0] * (locations + max_hours + max_days)    
        
        # Based on the input from state information, encode the list with 1 w.r.t its location, time and day.
        state_encoded[state[0]] = 1
        state_encoded[locations + int(state[1])] = 1
        state_encoded[locations + max_hours + int(state[2])] = 1
        
        return state_encoded


    # Use this function if you are using architecture-2 
#    def state_encod_arch2(self, state, action):
#         """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        
#        return state_encoded


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

        possible_actions_index = random.sample(range(1, (locations-1)*locations +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
       
        actions.append((0,0))

        return possible_actions_index, actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start2pickup_time = 0
        pickup2drop_time = 0
        
        start_loc = state[0]
        start_time = state[1]
        start_day = state[2]
        
        pickup_loc = action[0]
        drop_loc = action[1]
        
        if action == [0,0]:
            reward = -cost
        else:
            if start_loc != pickup_loc:
                start2pickup_time = Time_matrix[start_loc, pickup_loc, int(start_time), int(start_day)]
                pickup_time = start_time + start2pickup_time
                pickup_day = start_day
                
                if pickup_time >= max_hours:
                    pickup_time = pickup_time - max_hours
                    pickup_day = pickup_day + 1
                    if pickup_day >= max_days:
                        pickup_day = pickup_day - max_days
                        
                pickup2drop_time = Time_matrix[pickup_loc, drop_loc, int(pickup_time), int(pickup_day)]
                
                reward = (revenue * pickup2drop_time) - (cost * pickup2drop_time) - (cost * start2pickup_time) 
                    
            else:
                pickup2drop_time = Time_matrix[pickup_loc, drop_loc, int(start_time), int(start_day)]
                
                reward = (revenue * pickup2drop_time) - (cost * pickup2drop_time)
            
        return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        start_loc = state[0]
        start_time = state[1]
        start_day = state[2]
        
        pickup_loc = action[0]
        drop_loc = action[1]
        
        noRide_time = 0
        ride_time = 0
        ride2pick_time = 0
        pickup2drop_time = 0
        
        if pickup_loc == 0 and drop_loc == 0:
            noRide_time += 1
            
            next_loc = start_loc
            drop_time = noRide_time
            drop_day = start_day
            
            if drop_time >= max_hours:
                drop_time = drop_time - max_hours
                
                drop_day = drop_day + 1
                if drop_day >= max_days:
                    drop_day = drop_day - max_days
            
            self.total_ridetime = self.total_ridetime + noRide_time
            
        elif start_loc == pickup_loc:
            ride_time = Time_matrix[pickup_loc, drop_loc, int(start_time), int(start_day)]
            next_loc = drop_loc
            drop_time = start_time + ride_time
            drop_day = start_day
            
            if drop_time >= max_hours:
                drop_time = drop_time - max_hours
                
                drop_day = drop_day + 1
                if drop_day >= max_days:
                    drop_day = drop_day - max_days
            
            self.total_ridetime += ride_time
            
        else:
            ride2pick_time = Time_matrix[start_loc, pickup_loc, int(start_time), int(start_day)]
            pickup_time = start_time + ride2pick_time
            pickup_day = start_day
                
            if pickup_time >= max_hours:
                pickup_time = pickup_time - max_hours
                
                pickup_day = pickup_day + 1
                if pickup_day >= max_days:
                    pickup_day = pickup_day - max_days
                        
            pickup2drop_time = Time_matrix[pickup_loc, drop_loc, int(pickup_time), int(pickup_day)]
            drop_time = pickup_time + pickup2drop_time
            drop_day = pickup_day
            
            if drop_time >= max_hours:
                drop_time = drop_time - max_hours
                
                drop_day = drop_day + 1
                if drop_day >= max_days:
                    drop_day = drop_day - max_days
            
            next_loc = drop_loc
            
            next_time = ride2pick_time + pickup2drop_time
            self.total_ridetime += next_time
        
        if self.total_ridetime > self.maxhours2terminal:
            terminal = True
        else:
            terminal = False
        
        next_state = (int(next_loc), int(drop_time), int(drop_day))
        
        return next_state, terminal, int(self.total_ridetime)
    
    def isTerminalAchieved(self):
        if self.total_ridetime > self.maxhours2terminal:
            terminal = True
        else:
            terminal = False
        
        return terminal

    def reset(self):
        return self.action_space, self.state_space, self.state_init
