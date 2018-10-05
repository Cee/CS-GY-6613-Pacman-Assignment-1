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
        
        # Ensure initial state is not None
        if not state:
            return Directions.STOP
        # Ensure initial state will not be in a lost
        if state.isLose():
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP

        # Use FIFO queue to keep the search area, where we store
        # a path to the current state
        queue = []

        # Since there's a really low possibilty that state would be the
        # same in Pacman, I removed "explored" set here.
        # explored = Set()

        action = self.bfs(queue, state)
        return action

    def bfs(self, queue, init_state):
        # Get initial actions for initial state
        init_actions = init_state.getLegalPacmanActions()
        for action in init_actions:
            queue.insert(0, [(0, action, init_state)]) # (depth, prev_action, prev_state)

        while len(queue) != 0:
            path = queue[-1]
            (depth, prev_action, prev_state) = path[-1]
            curr_state = prev_state.generatePacmanSuccessor(prev_action)
            if curr_state == None:
                # Running out of `generatePacmanSuccessor`
                break
            if curr_state.isWin():
                return path[0][1]
            if curr_state.isLose():
                _ = queue.pop()
                continue
            # No win or lose, generate next actions
            curr_actions = curr_state.getLegalPacmanActions()
            for action in curr_actions:
                queue.insert(0, path + [(depth + 1, action, curr_state)])
            # Remove this node from the queue
            _ = queue.pop()

        if len(queue) == 0:
            return Directions.STOP
        # Still have elements
        pq = []
        for path in queue:
            action = path[0][1]
            depth = path[-1][0]
            state = path[-1][2]
            self.append(pq, (action, self.f(depth, state)))
        return self.pop(pq)[0]

    def f(self, depth, state):
        return depth + admissibleHeuristic(state);

    def append(self, pq, node):
        pq.append(node)

    def pop(self, pq):
        pq.sort(key = lambda node: -node[-1])
        return pq.pop()

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # Ensure initial state is not None
        if not state:
            return Directions.STOP
        # Ensure initial state will not be in a lost
        if state.isLose():
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP
        
        # Use LIFO queue / stack
        stack = []

        action = self.dfs(stack, state)
        return action

    def dfs(self, stack, init_state):
        # Get initial actions for initial state
        init_actions = init_state.getLegalPacmanActions()
        for action in reversed(init_actions):
            stack.append([(0, action, init_state)]) # (depth, prev_action, prev_state)

        while len(stack) != 0:
            path = stack[-1]
            (depth, prev_action, prev_state) = path[-1]
            curr_state = prev_state.generatePacmanSuccessor(prev_action)
            if curr_state == None:
                # Running out of `generatePacmanSuccessor`
                break
            if curr_state.isWin():
                return path[0][1]
            if curr_state.isLose():
                _ = stack.pop()
                continue
            # No win or lose, generate next actions
            curr_actions = curr_state.getLegalPacmanActions()
            for action in reversed(curr_actions):
                stack.append(path + [(depth + 1, action, curr_state)])
            # Remove this node from the stack
            _ = stack.pop()

        if len(stack) == 0:
            return Directions.STOP
        # Still have elements
        pq = []
        for path in stack:
            action = path[0][1]
            depth = path[-1][0]
            state = path[-1][2]
            self.append(pq, (action, self.f(depth, state)))
        return self.pop(pq)[0]

    def f(self, depth, state):
        return depth + admissibleHeuristic(state);

    def append(self, pq, node):
        pq.append(node)

    def pop(self, pq):
        pq.sort(key = lambda node: -node[-1])
        return pq.pop()

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # Ensure initial state is not None
        if not state:
            return Directions.STOP
        # Ensure initial state will not be in a lost
        if state.isLose():
            return Directions.STOP
        # Goal test for initial state, which is the solution
        if state.isWin():
            return Directions.STOP

        # f(n) = g(n) + h(n)
        # Use PriorityQueue
        pq = []

        action = self.astar(pq, state)
        return action

    def astar(self, pq, init_state):
        # Get initial actions for initial state
        init_actions = init_state.getLegalPacmanActions()
        g = 0                               # g: depth
        h = admissibleHeuristic(init_state) # h: heruistic(state)
        f = g + h
        for action in init_actions:
            # Tuple: (g, h, f, action, state)
            self.append(pq, [(g, h, f, action, init_state)])

        while len(pq) != 0:
            path = pq[-1]
            (depth, _, _, prev_action, prev_state) = path[-1]
            curr_state = prev_state.generatePacmanSuccessor(prev_action)
            if curr_state == None:
                # Running out of `generatePacmanSuccessor`
                # Current node has the minimum f(n)
                return path[0][-2]
            if curr_state.isWin():
                return path[0][-2]
            if curr_state.isLose():
                _ = pq.pop()
                continue
            g = depth + 1
            h = admissibleHeuristic(curr_state)
            f = g + h
            # No win or lose, generate next actions
            curr_actions = curr_state.getLegalPacmanActions()
            for action in curr_actions:
                self.append(pq, path + [(g, h, f, action, curr_state)])
            # Remove this node from the stack
            _ = pq.pop()

    def append(self, pq, node):
        pq.append(node)
        pq.sort(key = lambda node: -node[-1][2])