from game import getStartingBoard, makeMove, getValidMoves, score, printBoard, isOver, getNextState, getFen, copyBoard
from mcts import MCTS_Node
import threading
import queue
from random import choice
import json


def dump_data(root, name, epoch):
    lines = get_data(root)
    with open("training_data/" + name + ".json", "w") as f:
        data = {"epoch": epoch, "training": lines}
        f.write(json.dumps(data))

def get_data(node):
    lines = []
    fen = getFen(node.state)
    wins = node._results[1]
    losses = node._results[-1]
    total = node.n()
    node_line = [fen, [wins/total, losses/total]]

    if len(node.children) == 0:
        return [node_line]
    else:
        lines = [node_line]
        for i in [get_data(x) for x in node.children]:
            lines.extend(i)
        return lines

class Worker(threading.Thread):
    def __init__(self, q, data, t, *args, **kwards):
        self.q = q
        self.data = data
        self.t = t
        super().__init__(*args, **kwards)

    def run(self):
        while True:
            try:
                board = self.q.get(timeout=3)
                # print(board)
                n = MCTS_Node(board)
                print("Thread %s is starting training 1000 iterations on board %s"% (self.t, board))
                n.train(1000)
                self.data.extend(get_data(n))

                # evaluate_board(board, self.m1, self.m2)
            except queue.Empty:
                return
            
            self.q.task_done()

#Start game trainings
epoch = 10
for i in range(64):
    board = getStartingBoard()
    data = []
    
    #Iterate over each of the openings
    q = queue.Queue()
    for b in [getNextState(board, x[0], x[1]) for x in getValidMoves(board)]:
        q.put_nowait(b)
    for t in range(4):
        Worker(q, data, t).start()
    q.join()
    # print(len(data))
    # n = MCTS_Node(b)
    # n.train(10000)
    # data.extend(get_data(n))
        # print(get_data(n))

    e = epoch + i
    with open("training_data/checkpoint" + str(e) + ".json", "w") as f:
        data = {"epoch": epoch, "training": data}
        f.write(json.dumps(data))


# for i in n.children[-1].children:
#     print(i)
#     printBoard(i.state)

# print(n.best_child(c_param=0.))
# printBoard(n.best_child(c_param=0.).state)
# skipped = False
# while not isOver(b):
#     moves = getValidMoves(b)
#     print(moves)
#     m = choice(moves)
#     b = getNextState(b, m[0], m[1])
#     printBoard(b)

