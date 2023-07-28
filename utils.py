from matplotlib import pyplot as plt
import torch 
from DataLoaderForTorch import DataLoader_csv
from dtw import dtw
import numpy as np
from tqdm import tqdm
def plot_tensor(x,y,name):
    # x = x.item()
    # y = y.item()
    x0 = x[0]
    y0 = y[0]
    x0 = x0.to('cpu')
    y0 = y0.to('cpu')
    # x0 = torch.reshape(x0,x.shape[1])
    # y0 = torch.reshape(y0,y.shape[1])
    plt.plot(x0.detach().numpy(),'r')
    plt.plot(y0.detach().numpy(),'b')
    plt.show()
    plt.savefig('./pics/' + name + '.jpg')

def draw():
    x = torch.load('./newdatas/energy_CRR.data',map_location = 'cpu')
    y = torch.load('./newdatas/energy_None.data',map_location = 'cpu')
    print(x.shape,y.shape)
    for i,x0 in enumerate(x[-1]):
        plt.plot(x0.detach().numpy(),)
        if i == 3:break
    # for j,y0 in enumerate(y[-1]) :
    #     plt.plot(y0.detach().numpy(),)
    #     if j ==3 :break
        # print(y0)
    
    plt.savefig('./pics/datas.jpg')





def Dtw_of_datasets(rawdatas,newdatas,num:int):
    result = []
    for i ,datas in enumerate( newdatas):
        # 里面包含不同epsilon时的合成数据
        D = 0
        # dataset_processing = tqdm(datas)
        # dataset_processing.set_description('dataset_processing: ')
        dataset_processing = datas
        for j,x in enumerate(dataset_processing):
            d = float('inf')
            # single_data = tqdm(rawdatas)
            # single_data.set_description('single processing: ')
            single_data = rawdatas
            for y ,label in  single_data:
                t_d ,_,__,___ = dtw(x.view(-1).numpy(),
                                    y.view(-1).numpy(),
                                    lambda x,y : np.abs(x-y))
                if t_d < d :d = t_d
            D = D + d
            # print(f'D:{D}')
            if j == num:break
        # D = D / rawdatas.__len__()
        D = D / num
        print(f'已完成第{i}个数据集计算，D = {D}')
    result.append(D)
    return result

def analysis(num:int):
    CRR = torch.load('./newdatas/energy_CRR.data')
    Gaussian = torch.load('./newdatas/energy_Gaussian.data')
    No_privacy = torch.load('./newdatas/energy_None.data')
    print(f'loaded No privacy data')
    raw_data = DataLoader_csv('energy',1)
    print(f'loaded raw_data')
    r1 = Dtw_of_datasets(raw_data,No_privacy,num)
    print(r1)
    r2 = Dtw_of_datasets(raw_data,CRR,num)
    print(r2)
    r3 = Dtw_of_datasets(raw_data,Gaussian,num)
    print(r1)
    print(r2)
    print(r3)
    
if __name__ == '__main__':
    # draw()
    # x = [1,2,3,4,5,6,7]
    # y = [2,3,5,4,6]
    # d,_,__,path = dtw(x,y,lambda x,y : np.abs(x-y))
    # print(d)
    # print(path)
    # plt.plot(path[0],path[1])
    # plt.savefig('./pics/dtwtest.jpg')
    analysis(100)
