# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        
        # Ensure state is not None
        if not state:
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP

        # Use FIFO queue to keep the search area
        # where we store a path for returning the second element of 
        # the path array as the next step
        queue = []
        queue.insert(0, [(Directions.STOP, state)])

        path = self.bfs(queue, explored, Directions.STOP, [(Directions.STOP, state)])
        return path[1][0] if len(path) >= 2 else Directions.STOP

    def bfs(self, queue, explored, init_action, init_path):
        action = init_action
        path = init_path
        while len(queue) != 0:
            path = queue.pop()
            state = path[-1][1]
            if state.isLose():
                continue
            for action in state.getLegalPacmanActions():
                next_state = state.generatePacmanSuccessor(action)
                next_path = path + [(action, next_state)]
                if (next_state) and (next_path not in queue):
                    if next_state.isWin():
                        return next_path
                    queue.insert(0, next_path)
        return path

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # Ensure state is not None
        if not state:
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP
        
        # Use FILO/LIFO queue / stack
        stack = []
        stack.append([(Directions.STOP, state)])

        path = self.dfs(stack)
        return path[1][0] if len(path) >= 2 else Directions.STOP

    def dfs(self, stack):
        while len(stack) != 0:
            path = stack.pop()
            state = path[-1][1]
            for action in state.getLegalPacmanActions():
                next_state = state.generatePacmanSuccessor(action)
                next_path = path + [(action, next_state)]
                if (next_state):
                    if next_state.isWin():
                        return next_path
                    if next_state.isLose():
                        continue
                    stack.append(next_path)
        return path

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # Ensure state is not None
        if not state:
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP

        # f(n) = g(n) + h(n)
        # Use PriorityQueue
        pq = []
        start = ([state], [], 0, self.f(0, state)) # (states, actions, g, f = g + h)
        self.append(pq, start)

        while len(pq) != 0:
            node = self.pop(pq)
            s = node[0][-1] # current state
            d = node[-2] # depth
            for action in s.getLegalPacmanActions():
                next_state = s.generatePacmanSuccessor(action)
                if next_state:
                    if next_state.isLose():
                        continue
                    if next_state.isWin():
                        return node[1][0]
                    next_node = (node[0] + [next_state], node[1] + [action], d + 1, self.f(d + 1, next_state))
                    self.append(pq, next_node)
                else:
                    return node[1][0]

        return Directions.STOP

    def f(self, depth, state):
        return depth + admissibleHeuristic(state);

    def append(self, pq, node):
        pq.append(node)

    def pop(self, pq):
        pq.sort(key = lambda node: -node[-1])
        return pq.pop()