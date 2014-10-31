def cumsum(xs):
    acc = 0
    acc_list = [0]*len(xs)
    for i,x in enumerate(xs):
        acc += x
        acc_list[i] = acc
    return acc_list
