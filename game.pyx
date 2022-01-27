import sys

# from numpy import array
import numpy as np
from numpy.core.numeric import base_repr
cimport cython
from cpython cimport array
import array
import numpy

cdef int WHITE = 1
cdef int BLACK = -1

def getStartingBoard():
    return [[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 0, 0, 0, WHITE, BLACK, 0, 0, 0, 2], [2, 0, 0, 0, BLACK, WHITE, 0, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 0, 0, 0, 0, 0, 0, 0, 0, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], 1]

@cython.boundscheck(False)
@cython.wraparound(False)
def printBoard(board):
    mapping = {0: '   ', BLACK: '⚫ ', WHITE:'⚪ '}
    print("╔═══╦═══╦═══╦═══╦═══╦═══╦═══╦═══╗")
    for x in range(8):
        sys.stdout.write("║")
        for y in range(8):
            sys.stdout.write(mapping[board[0][x+1][y+1]])
            if y != 7:
                sys.stdout.write("║")
            else:
                print("║")
        if x != 7:
            print("╠═══╬═══╬═══╬═══╬═══╬═══╬═══╬═══╣")
    print("╚═══╩═══╩═══╩═══╩═══╩═══╩═══╩═══╝")

@cython.boundscheck(False)
@cython.wraparound(False)
def isValidMove(board, int mX, int mY):
    # print(board[0])
    # print(board[1])
    if board[0][mX][mY] != 0:
        return False
    
    cdef int curX = mX
    cdef int curY = mY    
    for d in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
        curX = curX + d[0]
        curY = curY + d[1]
        if board[0][curX][curY] == board[1] * -1:
            while True:
                if board[0][curX][curY] == 2:
                    break
                if board[0][curX][curY] == board[1]:
                    return True
                elif board[0][curX][curY] == 0:
                    break
                # cur = [cur[0] + d[0], cur[1] + d[1]]
                curX += d[0]
                curY += d[1]
        curX = mX
        curY = mY
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
def copyBoard(b):
    nB = []
    for x in range(10):
        nB.append([])
        for y in range(10):
            nB[x].append(b[0][x][y])
    return [nB, b[1]]

@cython.boundscheck(False)
@cython.wraparound(False)
def getFen(b):
    nB = []
    for x in range(1,9):
        for y in range(1,9):
            nB.append((b[0][x][y]+1)/(2))
        # nB.extend(b[x])
    nB.append(b[1])
    return nB

@cython.boundscheck(False)
@cython.wraparound(False)
def getValidMoves(board):
    acc = []
    # cdef int[:, :] acc = numpy.empty((64,2), dtype=int)
    # cdef int count = 0
    # cdef array.array acc = array.array
    for x in [1,2,3,4,5,6,7,8]:
        for y in [1,2,3,4,5,6,7,8]:
            if board[0][x][y] == 0:
                if isValidMove(board, x, y):
                    acc.append([x,y])
                    # acc[count, 0] = x
                    # acc[count, 1] = y
                    # count += 1
                
    return acc

@cython.boundscheck(False)
@cython.wraparound(False)
def isOver(board):
    return len(getValidMoves(board)) == len(getValidMoves([board[0], board[1]*-1])) == 0

@cython.boundscheck(False)
@cython.wraparound(False)
def makeMove(board, int mX, int mY):
    board[0][mX][mY] = board[1]
    cdef int curX = mX
    cdef int curY = mY   
    for d in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
        curX = curX + d[0]
        curY = curY + d[1]
        tiles = [[curX, curY]]
        if board[0][curX][curY] != 2:
            if board[0][curX][curY] == board[1] * -1:
                while True:
                    if board[0][curX][curY] == 2:
                        break
                    if board[0][curX][curY] == board[1]:
                        for t in tiles:
                            board[0][t[0]][t[1]] = board[1]
                        break
                    elif board[0][curX][curY] == board[1] * -1:
                        tiles.append([curX, curY])
                    elif board[0][curX][curY] == 0:
                        break
                    curX += d[0]
                    curY += d[1]
        curX = mX
        curY = mY
    board[1] *= -1
    return board

@cython.boundscheck(False)
@cython.wraparound(False)
def getNextState(b, int mX, int mY):
    board = copyBoard(b)
    board[0][mX][mY] = board[1]
    cdef int curX = mX
    cdef int curY = mY   
    for d in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
        curX = curX + d[0]
        curY = curY + d[1]
        tiles = [[curX, curY]]
        if board[0][curX][curY] != 2:
            if board[0][curX][curY] == board[1] * -1:
                while True:
                    if board[0][curX][curY] == 2:
                        break
                    if board[0][curX][curY] == board[1]:
                        for t in tiles:
                            board[0][t[0]][t[1]] = board[1]
                        break
                    elif board[0][curX][curY] == board[1] * -1:
                        tiles.append([curX, curY])
                    elif board[0][curX][curY] == 0:
                        break
                    curX += d[0]
                    curY += d[1]
        curX = mX
        curY = mY
    board[1] *= -1

    if len(getValidMoves(board)) == 0:
        board[1] *= -1
        return board
    return board

@cython.boundscheck(False)
@cython.wraparound(False)
def score(board):
    w = 0
    b = 0
    for x in range(8):
        for y in range(8):
            if board[0][x+1][y+1] == WHITE:
                w += 1
            elif board[0][x+1][y+1] == BLACK:
                b += 1
    return [w,b]