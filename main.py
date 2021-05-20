#CASE STUDY 1
#OPTIMIZING WAREHOUSE FLOWS 
#PROBLEM STATEMENT

''' We have a warehouse with like a maze and there is an automated machine that picks and delivers the
packages present there, the different places in maze has unique alphabets as their name with different
rankings. We have to make our MACHINE smart so as it'll pick the packages according to the priority order
and will move on the map by itself'''

#Importing Libraries
import numpy as np

#Setting parameters for alpha(learning factor) and gamma(discount factor)
gamma = 0.75
alpha = 0.9  #Low value = Slow Learning / High Value = Fast Learning

#PART 1 - DEFINING THE ENVIRONMENT

#Defining states
location_to_state = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'l':11}


#defining actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

#Defining rewards
rewards = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
                    [1,0,1,0,0,1,0,0,0,0,0,0],  
                    [0,1,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,1,0,0,0,0,0,0,0,1,0,0],
                    [0,0,1,0,0,0,1,1,0,0,0,0],
                    [0,0,0,1,0,0,1,0,0,0,0,1],
                    [0,0,0,0,1,0,0,0,0,1,0,0],
                    [0,0,0,0,0,1,0,0,1,0,1,0],
                    [0,0,0,0,0,0,0,0,0,1,0,1],
                    [0,0,0,0,0,0,0,1,0,0,1,0]])


#PART 2 - BUILDING THE AI SOLUTION IE Q LEARNING

#Making a mapping from the states to the locations
state_to_locations = {state:location for location, state in location_to_state.items()}

#Making the final function that will return the optimal route

def route(starting_location,ending_location):
    rewards_new = np.copy(rewards)
    ending_state = location_to_state[ending_location]
    rewards_new[ending_state, ending_state] = 1000
    
    #Initializing the Q values
    Q = np.array(np.zeros([12,12]))

    #Implementing Q Learning Process
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if rewards_new[current_state,j]>0:
                playable_actions.append(j)
                
        next_state = np.random.choice(playable_actions)
        TD = rewards_new[current_state,next_state] + gamma * Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
        Q[current_state,next_state] = Q[current_state,next_state] + alpha * TD
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_locations[next_state]
        route.append(next_location)
        starting_location = next_location
    print( route )

#PART 3 GOING INTO PRODUCTION
#Printing the final route
print('Route:')
route('A','K')