import math
import pdb
width = 500
height = 500

def getNextState(current_state, action):
    phip = current_state[4] + action
    vxp = -2*math.sin(math.radians(phip))
    vyp = -2*math.cos(math.radians(phip))
    xp = current_state[0] + vxp
    yp = current_state[1] + vyp
    next_state = [xp, yp, vxp, vyp, phip, 
            current_state[5], current_state[6]]
    return next_state

def isStateValid(state):
    isXValid = state[0] > 0 and state[0] < width
    isYValid = state[1] > 0 and state[1] < height
    return isXValid and isYValid

def distanceToGoal(state):
    x_distance = state[5] - state[0]
    y_distance = state[6] - state[1]
    d = math.sqrt(x_distance**2 + y_distance**2)
    return d

def reward(state):
    if(isStateValid(state)):
        max_d = math.sqrt(2)*width
        d = distanceToGoal(state)
        r = 1 - d/max_d
    else:
        r = 0
    return r

def tree_search(state, action, depth):
    next_state = getNextState(state, action)
    r = reward(next_state)
    if (depth == 0 or r == -10):
        return r
    else:
        rp2 = tree_search(next_state, 2, depth-1)
        r0  = tree_search(next_state, 0, depth-1)
        rm2 = tree_search(next_state, -2, depth-1)
        return 0.5*r + 0.5*(rp2 + r0 + rm2)/3

def policy(current_state):
    tree_depth = 5
    rp2 = tree_search(current_state, 2, tree_depth)
    r0 = tree_search(current_state, 0, tree_depth)
    rm2 = tree_search(current_state, -2, tree_depth)
    if(rp2 >= r0 and rp2 >= rm2):
        return 2
    elif (r0 >= rm2 and r0 >= rp2):
        return 0
    else:
        return -2

