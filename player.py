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

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 65)))
# model.add(Conv2D())
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model = load_model("./finished/2")


b = getStartingBoard()
skipped = False
while True:
    if b[1] == 1:
        printBoard(b)
        moves = getValidMoves(b)
        if len(moves) == 0:
            print("White Has No Moves, skipping turn")
            if skipped:
                break
            skipped = True
        else:
            bestMove = []
            bestScore = -1000
            for i in moves:
                score = model.predict([getFen(getNextState(b, i[0], i[1]))])
                print(score)
                if score[0][0] > bestScore:
                    bestScore = score[0][0]
                    bestMove = i
            print(bestMove)
            b = getNextState(b, bestMove[0], bestMove[1])
    else:
        moves = getValidMoves(b)
        printBoard(b)
        if len(moves) == 0:
            print("Black Has No Moves, Skipping Turn")
            if skipped:
                break
            skipped = True
        else:
            print("Valid moves: %s"% (' '.join([str(x) for x in moves])))
            m = [int(x) for x in input("Enter Move: ").split(' ')]
            b = getNextState(b, m[0], m[1])
print(score(b))

