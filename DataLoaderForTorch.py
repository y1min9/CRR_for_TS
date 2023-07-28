import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import arff
import numpy as np

class Datas_arff(Dataset):
    def __init__(self,name,transform=None,target_transform=None):
        super().__init__()
        self.data ,self.label = self.load_arff(name)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx:int):
        return self.data[idx] , self.label[idx]


    def load_arff(self,name):
        data,meta = arff.loadarff(name)
        X = []
        Y = []
        for row in data:
            x = []
            for point in row :
                # print(type(x))
                if type(point.item()) == type(0.1):
                    x.append(point)
                else:
                    y = int(point.item())
                    # print(type(y))
            X.append(x)
            Y.append(y)
        # print(f'len(y):{np.array(Y).shape[0]}')
        # length = len(X)
        X = np.array(X)
        X = X.reshape((X.shape[0],X.shape[1],1))
        return X ,Y 
class Datas_MIA(Dataset):
    def __init__(self,name,transform=None,target_transform=None) -> None:
        super().__init__()
        self.data = name
        # name 作为一个Tensor组成的列表
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) :
        return self.data[index],'this a label'
    
class Datas_csv(Dataset):
    def __init__(self,name,transform=None,target_transform=None):
        super().__init__()
        self.data  = self.load_CSV(name)
        # 这里控制一下数据规模，以期望更强的攻击
        # self.data = self.data[:9000]
        self.transform = transform
        self.target_transform = target_transform
        print(f'Datas_csv init ...  len = {self.__len__()}')
    
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx:int):
        return self.data[idx] ,'this is a label'
    
    def MinMaxScaler(self,data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return torch.tensor(numerator / (denominator + 1e-7))                  
    
    def load_CSV(self,name:str):    
        with open('./data/' + name + '.csv', 'r') as f:
            lines = f.readlines()

        # remove double quotes and split by comma
        X = [line.replace('"', '').strip().split(',') for line in lines]

        # convert to numpy array
        arr = np.array(X)
        # arr.astype(float)
        x = arr[1:,1:].astype(float)
        X = []
        for L in x:
            # X.append(self.MinMaxScaler(L))
            L = L.reshape(L.shape[0],1)
            # X = tensor_append(X,self.Normlize(L),only_merge=False)
            X = tensor_append(X,self.MinMaxScaler(L),only_merge=False)
            # X.append(L)
        # X = np.array(X)
        # X = X.reshape((X.shape[0],X.shape[1],1))
        return X
    
    def analysis(self):
        val = 0
        for x in self.data:
            val = x + val
        return val/self.__len__()
    def get_data(self):
        return self.data
    def Normlize(self,x):
        y = torch.tensor(x)
        mean = torch.mean(y)
        var = torch.var(y,unbiased=False)
        return (y - mean)/torch.sqrt(var)
    
    
def tensor_append(X,x, only_merge = True):
    # print(X,x)
    # print(f'X.shape:{X.shape}')
    # print(f'x.shape:{x.shape}')
    if only_merge:
        shape = []
    else:shape = [1]
    if x.shape == torch.Size([]):
        shape = shape + [1]  
    else:
        for _ in x.shape:
            shape.append(_)
    if X == []:
        # return x.reshape(shape)
        return x.reshape(shape)
    else:
        return torch.cat([X,x.reshape(shape)],dim = 0)
        # return torch.cat([X,x],dim = 0)

def DataLoader_MIA(name,batch_size = 128,shuffle = True):
    datas = Datas_csv(name,transform=transforms.ToTensor())
    data_itt = DataLoader(datas,1,shuffle)
    length = data_itt.__len__()
    # 为保证攻击效果，这里缩小了数据规模
    train_point = int(length * 0.49)
    attak_point = int(length * 0.98)
    datas_victim = []
    datas_attcak = []
    datas_test = []
    print(f'DataLoader_MIA ...')
    for i,(x,label) in  enumerate( data_itt):
        # print(f'type(x):{type(x)}')
        # print(f'x:{x}')
        if i == 0:
            print(f'Preparing victims\' training dataset ... ')
        elif i == train_point:
            print(f'Preparing attacker\' training dataset ... ')
        elif i == attak_point:
            print(f'Preparing test dataset ... ')
 
 
        if i < train_point:
            datas_victim = tensor_append(datas_victim,x)
            if i < length - attak_point:
                datas_test = tensor_append(datas_test,x)
        elif i >= train_point and i < attak_point:
            datas_attcak = tensor_append(datas_attcak,x)
            # print(f'data_victim.shape:{datas_victim.shape}')
            # print(f'data_attack.shape:{datas_attcak.shape}')
            # break
        else:
            datas_test = tensor_append(datas_test,x)
    print(f'finished MIA dataloading ...')
    print(f'{datas_victim.shape},{datas_attcak.shape},{datas_test.shape}')
    
    return DataLoader(Datas_MIA(datas_victim) ,batch_size,shuffle),DataLoader(Datas_MIA(datas_attcak),batch_size,shuffle),DataLoader(Datas_MIA(datas_test),1,False)
    # 被攻击方，攻击方的训练数据，测试数据顺序返回
            


    
def DataLoader_csv(name,batch_size=128,shuffle=True):
    
    datas = Datas_csv(name,transform= transforms.ToTensor())
    return DataLoader(datas,batch_size,shuffle)

def DataLoader_arff(name_,batch_size=128,shuffle=True):
    name = './data/' + name_ + '_TEST.arff'
    # datas = Datas_arff('./data/' + name+'_'+train_or_test)
    datas = Datas_arff(name,transform = transforms.ToTensor())
    # print(f'数据集大小： {datas.__len__()}')
    return DataLoader(datas,batch_size,shuffle)



if __name__ == '__main__':

    X,Y,Z = DataLoader_MIA('energy',batch_size=128)
    source = DataLoader_csv('energy',batch_size=128)
    print(f'size:{X.__len__()},{Y.__len__()},{Z.__len__()},{source.__len__()}')

    # Y = Datas_csv('energy')
    # YY = DataLoader_csv('energy',batch_size=3,shuffle = False)
    # for x,label in YY :
    #     print(x,label)
    #     print(x.shape)
    #     break
    
    # X = DataLoader_arff('Crop',6)
    # print(X.__len__())
    # for x,y in X:
    #     print(f'type(x):{type(x)}')
    #     print(f'type(x[0]):{type(x[0])}')
    #     # print(f'x.shape:{x.shape}')
    #     # print(x)
    #     # x = torch.stack(x)
    #     # x = np.array(x)
    #     print(f'x.shape:{x.shape}')
    #     break


    # load_arff('./data/Crop_TEST.arff')
    # datas = Datas_arff('Crop')
    # print(f'len:{dataloader.__len__()}')
    # print(f'getitem:{dataloader.__getitem__(3)}')
    # train_data = DataLoader(datas,batch_size = 10,shuffle = False)
    # print(train_data.next())

    # i = 0
    # for X ,Y in train_data:
    #     print(X)
    #     print(Y)
    #     i = i + 1
    #     if i == 10 :
    #         break
    # for x,y in dataloader:
    #     if y == i:
    #         print(x,y)
    #         i = i + 1
    # print(f'count:{i}')
