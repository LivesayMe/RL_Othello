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
import numpy as np
import json

import threading
import queue
# from server.game.game import getStartingBoard


def agent(board, player, moves):
    # m = randint(0, moves.shape[0]-1)
    # return [moves[m, 0], moves[m, 1]]
    return choice(moves)



def playGame():
    b = getStartingBoard()
    cP = 1
    turn = 1
    skipped = False
    while True:
        moves = getValidMoves(b, cP)
        if len(moves) > 0:
            move = agent(b, cP, moves)
            b = getNextState(b, move[0], move[1], cP)
            printBoard(b)
        else:
            if skipped:
                break
            else:
                skipped = True
        cP *= -1

import timeit
# b = getStartingBoard()
# moves = getValidMoves(b, 1)
# print(moves.shape[0])
# print(timeit.timeit(lambda: playGame(), number=100)/100)

discount_factor = 0.95
eps = 0.5
eps_decay_factor = 0.999
num_episodes=500


# model = Sequential()
# model.add(InputLayer(batch_input_shape=(1, 64)))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(2, activation='linear'))
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# # mypath='./checkpoints/'
# # onlyfiles = [f for f in listdir(mypath)]
# # if len(onlyfiles) > 0:
# #     print(onlyfiles[-1])
# #     model = load_model(mypath + onlyfiles[-1])

# co = 0
# wins = []
# for i in range(num_episodes):
#     print("Starting epoch: %s"% (i))

#     b = getStartingBoard()
#     cP = 1
#     eps *= eps_decay_factor
#     done = False
#     skipped = False
#     count = 0
#     while not done:
#         moves = getValidMoves(b, cP)

#         if len(moves) > 0:
#             if np.random.random() < eps:
#                 action = choice(getValidMoves(b, cP))
#             else:
#                 best = -10000
#                 bM = []
#                 for i in moves:
#                     # print(b)
#                     # printBoard(b)
#                     nS = getNextState(b, i[0], i[1], cP)
#                     # print(nS)
#                     w = model.predict([getFen(nS)])[0][0]
#                     # print(w)
#                     if w > best:
#                         best = w
#                         bM = i
#                 action = bM


#             new_state = getNextState(b, action[0], action[1], cP)
#             s = score(new_state)
#             reward = s[0] - s[1]
#             if isOver(new_state):
#                 if reward > 0:
#                     reward = 10000
#                 else:
#                     reward = -10000

#             best = -10000
#             bestState = []
#             nMoves = getValidMoves(new_state, cP*-1)
#             if len(nMoves) > 0:
#                 for i in getValidMoves(new_state, cP*-1):

#                     #Black is trying to maximize now
#                     nS = getNextState(new_state, i[0], i[1], cP*-1)
#                     w = model.predict([getFen(nS)])[0][1]
#                     # print(w)
#                     if w > best:
#                         best = w
#                         bestState = nS
#                 if isOver(bestState):
#                     if sum(score(bestState)) > 0:
#                         target = 10000
#                     else:
#                         target = -10000
#                 else:
#                     target = reward + discount_factor * np.max(model.predict([getFen(bestState)]))
                
#             else:
#                 target = reward + discount_factor * 5

#             target_vector = model.predict([getFen(b)])[0]
#             target_vector[0] = target

#             model.fit(np.array([getFen(new_state)]), np.reshape(target_vector, [-1,2]), epochs=1, verbose=0)
#             b = copyBoard(new_state)
#             co += reward
#         else:
#             skipped = True
        
#         cP *= -1
#         moves = getValidMoves(b, cP)
#         if len(moves) > 0:
#             move = choice(moves)
#             b = getNextState(b, move[0], move[1], cP)
#         else:
#             if skipped:
#                 done = True
#         cP *= -1
        
#         count += 2
#         print("Score after round %s: %s"% (count, score(b)))
#         printBoard(b)
#         if done:
#             print("Score: %s"% co)
#             wins.append(co)
#             co = 0

#     # if i % 5 == 0:
#     #     model.save('./checkpoints/epoch-' + str(i))
# model.save('./finished/1')
# import matplotlib.pyplot as plt
# plt.plot(range(num_episodes), wins)

def train_from_checkpoint(checkpoint):
    training_data = []
    with open("./training_data/checkpoint" + str(checkpoint) + ".json") as f:
        training_data = json.loads(f.read())["training"]


    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 65)))
    # model.add(Conv2D())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    x_train = []
    y_train = []
    print(len(training_data))
    for i in training_data:
        # print(i)
        x_train.append(i[0])
        y_train.append(i[1])


    print("Training Data Loaded Into RAM, Starting Backprop")
    print(len(x_train))
    model.fit(np.array(x_train), np.array(y_train), epochs = 10, verbose=2)
    model.save('./finished/' + str(checkpoint))


def evaluate_board(board, m1, m2):

    results = {1: 0, 0: 0, -1: 0}
    for new_model in [True, False]:
        b = copyBoard(board)
        #Start the game loop
        skipped = False
        while True:
            #White's turn
            if b[1] == 1:
                moves = getValidMoves(b)
                #White has no valid moves
                if len(moves) == 0:
                    if skipped:
                        break
                    skipped = True
                    b[1] *= -1
                else:
                    bestMove = []
                    bestScore = -1000
                    for i in moves:
                        #Is the new model playing white?
                        if new_model:
                            curScore = m1.predict([getFen(getNextState(b, i[0], i[1]))])
                        else:
                            curScore = m2.predict([getFen(getNextState(b, i[0], i[1]))])

                        if curScore[0][0] > bestScore:
                            bestScore = curScore[0][0]
                            bestMove = i
                    b = getNextState(b, bestMove[0], bestMove[1])
            else:
                #Black's Turn
                moves = getValidMoves(b)
                if len(moves) == 0:
                    if skipped:
                        break
                    skipped = True
                    b[1] *= -1
                else:
                    bestMove = []
                    bestScore = -1000
                    for i in moves:
                        if new_model:
                            curScore = m1.predict([getFen(getNextState(b, i[0], i[1]))])
                        else:
                            curScore = m2.predict([getFen(getNextState(b, i[0], i[1]))])
                        if curScore[0][1] > bestScore:
                            bestScore = curScore[0][1]
                            bestMove = i
                    b = getNextState(b, bestMove[0], bestMove[1])
        sc = score(b)
        #Store the result, 0 is tie, 1 is white, 0 is black 
        if sc[0] == sc[1]:
            results[0] += 1
        elif sc[0] > sc[1]:
            results[1 if new_model else -1] += 1
        else:
            results[-1 if new_model else 1] += 1
    return results

class Worker(threading.Thread):
    def __init__(self, q, model1path, model2path, *args, **kwards):
        self.q = q
        self.m1 = load_model(model1path)
        self.m2 = load_model(model2path)
        super().__init__(*args, **kwards)

    def run(self):
        while True:
            try:
                board = self.q.get(timeout=3)
                # print(board)
                print(evaluate_board(board, self.m1, self.m2))
            except queue.Empty:
                return
            
            self.q.task_done()

def compare_performance(n1, n2, layer_count, win_threshold):
    model1 = "./finished/" + str(n1)
    model2 = "./finished/" + str(n2)
    
    board = getStartingBoard()

    boards = [board]
    for i in range(layer_count):
        new_boards = []
        for q in boards:
            for m in getValidMoves(q):
                new_boards.append(getNextState(q, m[0], m[1]))
        boards = new_boards
    # print(boards)
    print("Comparing agents %s and %s across %s openings"% (n1, n2, len(boards)))
    # results = {1: 0, 0: 0, -1: 0}


    #Iterate over each of the openings
    q = queue.Queue()
    for b in boards:
        q.put_nowait(b)
    for _ in range(10):
        Worker(q, model1, model2).start()
    q.join()
        
    
    # total_games = len(boards)
    # print(results[1]/total_games)
    # print(results[-1]/total_games)


# compare_performance(1, 2, 2, 1)

for i in range(64):
    train_from_checkpoint(i+10)

