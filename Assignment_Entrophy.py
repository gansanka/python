import random
import numpy as np

_Ilist = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
random.shuffle(_Ilist)
print('list : ',_Ilist)

balls_left = _Ilist[:10]
print('balls_left : ' , balls_left)
balls_right = _Ilist[10:]
print('balls_right : ', balls_right)


def calcEntropht(_given):
    count = len(_given)
    pos_1len = len(list(filter(lambda x: x == 1, _given)))
    pos_0len = len(list(filter(lambda x: x == 0, _given)))
    entrop = -((pos_1len/count * np.log2(pos_1len/count)) + (pos_0len/count * np.log2(pos_0len/count)))
    print("Entropy value : ", entrop)

calcEntropht(balls_left)

# entropy of a fair dice
dice = -(6*(1/6)*np.log2(1/6))
print('entropy of a fair dice : ',dice)