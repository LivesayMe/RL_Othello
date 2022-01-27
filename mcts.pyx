import numpy as np
from collections import defaultdict
import game

from game import getStartingBoard, makeMove, getValidMoves, score, printBoard, isOver, getNextState, getFen, copyBoard
from random import choice
from keras.models import Sequential
from keras.models import load_model
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Conv2D
from os import listdir
from os.path import isfile, join
import json

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 65)))
# model.add(Conv2D())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model = load_model("./finished/1")


class MCTS_Node():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._results[0.5] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return 
    
    def __str__(self):
        return f"Action: {self.parent_action}\nVisited: {self._number_of_visits}\nWin/Loss:{self._results[1]}/{self._results[-1]}"
    
    def untried_actions(self):
        self._untried_actions = game.getValidMoves(self.state)
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins-losses
    
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = game.getNextState(self.state, action[0], action[1])
        child_node = MCTS_Node(next_state, parent = self, parent_action = action)
        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self):
        return game.isOver(self.state)
    
    def rollout(self):
        current_rollout_state = self.state
        while not game.isOver(current_rollout_state):
            possible_moves = game.getValidMoves(current_rollout_state)
            action = self.rollout_policy(current_rollout_state, possible_moves)
            current_rollout_state = game.getNextState(current_rollout_state, action[0], action[1])
        s = game.score(current_rollout_state)
        return .5 if s[0] == s[1] else (1 if s[0] > s[1] else -1)
    
    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt(self.n())/c.n() for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, state, possible_moves):
        # bestMove = []
        # bestScore = -1000
        # for m in possible_moves:
        #     score = model.predict([getFen(getNextState(state, m[0], m[1]))])
        #     if score[0][0] > bestScore:
        #         bestScore = score[0][0]
        #         bestMove = m
        return choice(possible_moves)
        # return bestMove

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def train(self, simulation_no):
        for i in range(simulation_no):
            # print("Game %s"% (i))
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

    def best_action(self):
        return self.best_child(c_param=0.)