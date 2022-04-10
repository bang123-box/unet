import os

def path():
    path = r'D:\pythonwork\深度学习\第二次\msrc2_seg'
    train_p = 'Train.txt'
    valid_p = 'Validation.txt'
    test_p = 'Test.txt'
    train = []
    with open(os.path.join(path, train_p), 'r') as f:
        i = 0
        for line in f:
            train.append(line[:-1])

    validation = []
    with open(os.path.join(path, valid_p), 'r') as f:
        i=0
        for line in f:
            validation.append(line[:-1])
            i = i+1
    test = []
    with open(os.path.join(path, test_p), 'r') as f:
        for line in f:
            test.append(line[:-1])
    return (train, validation)

