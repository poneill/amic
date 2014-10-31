from time import time

stack = []
def tic(msg):
    global stack
    timestamp = time()
    stack.append((msg,timestamp))

def toc():
    global stack
    msg,timestamp = stack.pop()
    print "[%s: %1.2f]" % (msg,time() - timestamp)
