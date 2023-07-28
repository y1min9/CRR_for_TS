from typing import Any
import numpy as np
from torch.autograd import Function


class CRR(Function):
    
    @staticmethod
    def forward(ctx: Any,input,p) -> Any:
        if 0.55 >= np.random.rand():
            return input 
        else:
            return 1 - input
    @staticmethod
    def backward(ctx: Any,grad_output) -> Any:
        return  grad_output,None

def epsilon(p):
    return (2*p -1) * np.log(p/(1-p))


def CRR_Itt(epsilons,p):
    itt = []
    for eps in epsilons:
        itt.append(int(eps//epsilon(p)))
    return itt

def check_privacy(itt:int, iterations:list):
    if itt in iterations:
        return True
    else :
        return False
def check_privacy_DPSGD(epsilon,epsilon_0,epsilons:list):
    for x in epsilons:
        # if x-epsilon <= epsilon_0 and epsilon < 0:
        # if x > epsilon_0 and x < epsilon: 
        if x < epsilon and epsilon_0 < x:
            return True
    return False

def DP_dis(p,Y):
    if p >= np.random.rand():
        return Y
    else:
        return 1-Y




if __name__ == '__main__':
    import torch
    crr= CRR
    x = torch.arange(4.0,requires_grad= True)
    print(x)
    y = 2 * torch.dot(x,x)
    print(y)
    z = crr.apply(y,0.55)
    # y.backward()
    print(f'x.grad:{x.grad}')
    z.backward()
    print(f'x.grad:{x.grad}')
    # print(f'y:{y}')
    # print(torch.autograd.gradcheck(crr.apply,xx),eps=1e-6, atol=1e-4)
        # print(epsilon(0.55))
    # print(CRR_Itt([1,2,4,8],0.55))
    # iterations = CRR_Itt([1,2,4,8],0.55)
    # print(check_privacy(12,iterations))
    # print(check_privacy(199,iterations))