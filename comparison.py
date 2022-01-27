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

model2 = load_model("./finished/2")


game_count = 10
win_count = 0
board = getStartingBoard()
for b in [getNextState(board, x[0], x[1]) for x in getValidMoves(board)]:
    skipped = False
    while True:
        if b[1] == 1:
            printBoard(b)
            print(score(b))
            moves = getValidMoves(b)
            if len(moves) == 0:
                print("White Has No Moves, skipping turn")
                if skipped:
                    break
                skipped = True
                b[1] *= -1
            else:
                bestMove = []
                bestScore = -1000
                for i in moves:
                    curScore = model.predict([getFen(getNextState(b, i[0], i[1]))])
                    if curScore[0][0] > bestScore:
                        bestScore = curScore[0][0]
                        bestMove = i
                print("White Win Confidence: %s"% (bestScore))
                b = getNextState(b, bestMove[0], bestMove[1])
        else:
            moves = getValidMoves(b)
            printBoard(b)
            print(score(b))
            if len(moves) == 0:
                print("Black Has No Moves, Skipping Turn")
                if skipped:
                    break
                skipped = True
                b[1] *= -1
            else:
                bestMove = []
                bestScore = -1000
                for i in moves:
                    curScore = model2.predict([getFen(getNextState(b, i[0], i[1]))])
                    if curScore[0][1] > bestScore:
                        bestScore = curScore[0][1]
                        bestMove = i
                print("Black Win Confidence: %s"% (bestScore))
                b = getNextState(b, bestMove[0], bestMove[1])
    sc = score(b)
    print(sc)
    win_count += 1 if sc[0] > sc[1] else 0
print(win_count)