import matplotlib.pyplot as plt
import numpy as np

def list_to_numbers(l):
    left = 0; right = 0
    ret = []
    while (left < len(l)):
        if l[left] == ' ' or l[left] == '\t':
            left += 1
        else:
            right = left
            while right < len(l) and l[right] != ' ' and l[right] != '\t':
                right += 1
            # make l[left : right] to a number
            s = ''.join(l[left : right])
            num = float(s)
            ret.append(num)
            left = right + 1
    return ret

'''read train loss and eval loss from txt files'''
def draw_loss(loss_path, eval_path):
    with open(loss_path, 'r') as f:
        lossList = f.readline()
        lossList = list_to_numbers(lossList)
        '''even index is step, odd index is loss'''
        steps = lossList[::2]; lossList = lossList[1::2]
    with open(eval_path, 'r') as f:
        evalList = f.readline()
        evalList = list_to_numbers(evalList)
        evalSteps = evalList[::2]; evalLoss = evalList[1::2]
    a = np.arange(0, 12, 0.5)
    plt.yticks(a)
    plt.plot(steps, lossList, label='train loss')
    plt.plot(evalSteps, evalLoss, label='eval loss')
    plt.legend()
    plt.show()
        
draw_loss('train/2024_8_8_22_37/loss.txt', 'train/2024_8_8_22_37/eval.txt')
    