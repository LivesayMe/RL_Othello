from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

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

app = Flask(__name__)
cors = CORS(app)
run_with_ngrok(app) 

def evaluate_board(board, m1, m2):
    logs = []
    results = {1: 0, 0: 0, -1: 0}
    for new_model in [True]:
        log = []
        b = copyBoard(board)
        #Start the game loop
        skipped = False
        while True:
            printBoard(b)
            log.append(b)
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
        logs.append(log)
        #Store the result, 0 is tie, 1 is white, 0 is black 
        if sc[0] == sc[1]:
            results[0] += 1
        elif sc[0] > sc[1]:
            results[1 if new_model else -1] += 1
        else:
            results[-1 if new_model else 1] += 1
    return [results, logs]

@app.route("/compare")
def compare():
    m1 = load_model("./finished/" + request.args.get("m1"))
    m2 = load_model("./finished/" + request.args.get("m2"))
    r = evaluate_board(getStartingBoard(), m1, m2)
    return jsonify({"score": r[0], "logs": r[1]})

app.run()