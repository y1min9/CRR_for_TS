import torch
import torch.nn as nn
from sklearn.metrics import roc_curve,auc
# from matplotlib import pyplot as plt
from DP_tgan_torch import set_models_LSTM,tgan_pure
# from DP_tgan_torch import tgan
from DataLoaderForTorch import DataLoader_MIA,tensor_append
from dtw import dtw
import numpy as np
# 1.攻击思路：被攻击模型生成一个大小为k的数据集G_v
# 2.校准模型生成一个同样大小的数据集G_cal
# 3.给定一个待判断样本x,计算Loss(x,x_hat),x_hat in G_v的最小时的Loss
# 4.同时对x_hat in G_cal再计算一遍，求二者的差，将这个差值作为判断依据与预先设置的好的阈值比较。
# 5.由于受阈值取值影响较大，可以借助AUCROC选择最佳取值


class Loss(nn.Module):
    def __init__(self, LAMBDAs, netG=torch.zeros(3,3), distance = 'l2', if_norm_reg=False, z_dim=28):
        # 非图像数据，不考虑l2
        super(Loss, self).__init__()
        self.distance = distance
        # self.lpips_model = ps.PerceptualLoss()
        self.netG = netG
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim
        self.LAMBDAs = LAMBDAs
        
        # LAMBDAs用于平衡3项

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            # self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2)
            self.loss_lpips_fn = lambda x, y: 0.
        elif distance == 'dtw':
            print('Use distance: dtw')
            # self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2)
            self.loss_lpips_fn = lambda x, y: 0.
            self.loss_l2_fn = self.loss_dtw
        
        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
    def loss_dtw(self,x,y):
        l,_,__,___ = dtw(x.view(-1).numpy(),
                         y.view(-1).numpy(),
                         lambda x,y:np.abs(x-y))
        return l
    def forward(self, z , x_hat , x_gt):
        # self.x_hat = self.netG(z)
        # 这里修改了x_hat，直接传入x_hat ，而不需模型计算G(z)
        # z是一个具体的噪声
        self.x_hat = x_hat
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.LAMBDAs[0] * self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = self.LAMBDAs[1] * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            # z_ = z.view(-1, self.z_dim)
            #
            # norm = torch.sum(z_ ** 2, dim=1)
            norm = torch.sum(z ** 2)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += self.LAMBDAs[2] * norm_penalty

        return self.vec_loss
    
def get_loss(LAMBDA,is_white_box,test_data,noise,x_hats):
    # test_data 是直接读取的数据，是带标签的
    loss = Loss(LAMBDAs=LAMBDA,if_norm_reg = is_white_box)
    loss.eval()
    outs = []
    print(f'get_loss ...')
    print(f'Lambda:{LAMBDA},test_data:{test_data.__len__()},noise:{noise.__len__()},x_hats:{x_hats.__len__()}')
    for x_test,label in test_data:
        out = torch.tensor(-1)
        for n,x_hat in zip(noise,x_hats):
            _ = loss(n,x_hat,x_test)
            # print(f'out :{out}')
            # print(f'shape of out :{out.shape}')
            if torch.equal(out,torch.tensor(-1)):
                out = _
                # print(f'init out:{out}')
            elif _.item() < out.item():
                out = _
                # print(f'update out:{out}')
        
        outs = tensor_append(outs,out,only_merge=False)
    return outs

def get_matrix(l_cal,fig_label,fig_name):
    from matplotlib import pyplot as plt
    labels = torch.ones(l_cal.shape)
    labels[labels.shape[0]/2:] = 0
    fpr ,tpr , threshholds = roc_curve(labels,l_cal)

    plt.plot(fpr,tpr,lw = 1.5,label = fig_label)
    plt.xlabel("FPR",fontsize=15)
    plt.ylabel("TPR",fontsize=15)
    plt.title("ROC " + fig_name)
    plt.legend(loc="lower right")
    plt.show()
 
    # True_positive = 0
    # False_positive = 0
    # True_negetive = 0
    # False_negetive = 0
    # # shape是两位，例如[80,1]
    # for i,l in enumerate(l_cal):
    #     # l_cal < eps 表示推断在训练集中;i<pos表示在训练集中
    #     if l_cal < eps and i < pos:
    #         True_positive = True_positive + 1
    #     elif l_cal >= eps and i < pos:
    #         False_negetive = False_negetive + 1
    #     elif l_cal < eps and i >= pos:
    #         False_positive = False_positive + 1
    #     elif l_cal >= eps and i >= pos:
    #         True_negetive = True_negetive + 1
 


def attack_experiment(parameters,tgan):
    from matplotlib import pyplot as plt
    
    Loader_victim,loader_attcack,loader_test = DataLoader_MIA(parameters['dataset'],batch_size=parameters['batch_size'])
    # loader_test 前一半是训练集中的数据,后一半不在训练集中
    for x,label in Loader_victim:
        print(f'type(x) in Loader_victim:{type(x)}')
        print(f'shape(x) in Loader_victim:{x.shape}')
        break
    print(f'loader_test.__len__():{loader_test.__len__()}')
    datas_attack,noises_attack = tgan_pure(loader_attcack,parameters)
    print(f'datas_attack.shape:{datas_attack.shape}')
    
    datas_victim,noises_victim = tgan(Loader_victim,parameters)
    print(f'datas_vitim:{datas_victim.shape}')
    
    i = 0
    # calibration attack ____________________________________________________________
    data_attack,noise_attack = datas_attack[0],noises_attack[0]
    USE_NORM = True
    calibration_white_loss = get_loss(parameters['MIA_lambda'], USE_NORM, loader_test, noise_attack, data_attack)
    print(f'type of calibration_white_box_loss : {type(calibration_white_loss)}')
    print(f'shape of calibration_white_box_loss : {calibration_white_loss.shape}')
    #-------------------------------------------------------------------------------- 
    epsilons = parameters['epsilons']
    print(f'MIA epsilons:{epsilons}')
    for data_victim ,noise_victim in zip(datas_victim, noises_victim):#对应了不同epsilon
        # 这里注意Loss里面的dim，z_dim需要修改
        print(f'data_victim :{data_victim.shape}')
        print(f'noise_victim :{noise_victim.shape}')
        white_box_loss = get_loss(parameters['MIA_lambda'], USE_NORM, loader_test, noise_victim, data_victim)
        print(f'type of white_box_loss : {type(white_box_loss)}')
        print(f'shape of white_box_loss : {white_box_loss.shape}')
        

        L_white = white_box_loss - calibration_white_loss
        
        print(f'L.shape:{L_white.shape}')
        labels = torch.ones(L_white.shape)
        labels[int(labels.shape[0]/2):] = 0
        fpr ,tpr , threshholds = roc_curve(labels,L_white)
        AUC = auc(fpr,tpr)
        if parameters['Mech'] == 'None':
            print('Mech = None')
            plt.plot(fpr,tpr,lw = 1.5,label = f'epsilon = + inf, AUC={round(AUC,3)}' )
        else:
            plt.plot(fpr,tpr,lw = 1.5,label = f'epsilon = {epsilons[i]}, AUC={round(AUC,3)}' )
        i = i + 1
    plt.xlabel("FPR",fontsize=15)
    plt.ylabel("TPR",fontsize=15)
    plt.title("ROC of "+ parameters['Mech']  )
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('./pics/RoC_'+ parameters['Mech']  +'.jpg')
    plt.close()
    return
       
                
def MIA_from_data():
        # energy_CRR = torch.load('./newdatas/energy_CRR.data')
        energy_None = torch.load('./newdatas/energy_None.data')
        # print(energy_CRR[0])
        print(energy_None[0])

        # noises_CRR = torch.load('./test/noise_energy_CRR.data')
        # noises_None = torch.load('./test/noise_energy_None.data')
        # print(energy_None[0][0])
        # print(noises_CRR[0][0])
        # print(noises_None[0][0])
        
        # loss = get_loss([1.0,0,0.5],True,energy_None[0],noises_CRR[0],energy_CRR[0])
        # print(loss)

        # Loader_victim,loader_attcack,loader_test = DataLoader_MIA('energy')
        # for x,lable in loader_test:
        #     print(x)
        #     print(x.shape)
        #     x = x.view(-1)
        #     x = x.to('cpu').numpy()
        #     plt.plot(x)
        #     plt.savefig('./pics/norm.png')
        #     break


if __name__ =='__main__':
    MIA_from_data()
    