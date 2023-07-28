import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
from opacus.layers import DPGRU, DPLSTM, DPRNN
import datetime
from DataLoaderForTorch import DataLoader_arff
import utils
import DP
from DP import CRR
NAME =  'train_' +  str(datetime.datetime.now()) + '.txt'
LENGTH = 0

def MinMaxScaler(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val


class LSTM_FC(nn.Module):
    def __init__(self,input_size,hidden_nodes,output_size,batch_size,RNN_depth = 3):
        super().__init__()
        self.hidden_size = hidden_nodes
        self.output_size = output_size
        self.batch_size = batch_size
        self.RNN_depth = RNN_depth
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_nodes,num_layers = RNN_depth,batch_first= True)
        self.FC = nn.Sequential(
            nn.Linear(in_features = hidden_nodes , out_features = output_size),
            # nn.LeakyReLU()
            nn.Tanh()
            # nn.Sigmoid()
            # MSE的输入最好是有界
        )
    
    def forward(self,input):
        # h0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        # c0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        output, _ = self.lstm(input)
        # print(f'shape of output(after lstm):{output.shape}')
        output = self.FC(output)
        return output

class LSTM_decoder(nn.Module):
    def __init__(self,input_size,hidden_nodes,output_size,batch_size,RNN_depth = 3):
        super().__init__()
        self.hidden_size = hidden_nodes
        self.output_size = output_size
        self.batch_size = batch_size
        self.RNN_depth = RNN_depth
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_nodes ,num_layers = RNN_depth,batch_first= True)
        self.FC = nn.Sequential(
            nn.Linear(in_features = hidden_nodes , out_features = output_size),
            nn.Sigmoid() 
        )
    
    def forward(self,input):
        # h0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        # c0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        output, _ = self.lstm(input)
        # print(f'output:{output.shape}')
        output = self.FC(output)
        return output



class Decoder_attention(nn.Module):
    def __init__(self,input_size,hidden_nodes,output_size,batch_size,depth = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.outputsize = output_size
        self.batch_size = batch_size
        self.depth = depth
        self.attention = nn.TransformerDecoderLayer(d_model = self.input_size,nhead = 2,
                                               dim_feedforward = self.input_size * 4,
                                               batch_first = True,
                                               norm_first = True
                                               )
    def forward(self,input):
        return self.attention(input)

class Discriminator_LSTM(nn.Module):
    def __init__(self,input_size,hidden_nodes,output_size,batch_size,RNN_depth = 3):
        super().__init__()
        self.hidden_size = hidden_nodes
        self.output_size = output_size
        self.batch_size = batch_size
        self.RNN_depth = RNN_depth
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_nodes ,num_layers = RNN_depth,batch_first= True)
        self.FC = nn.Sequential(
            nn.Linear(in_features = hidden_nodes , out_features = output_size),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        # h0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        # c0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        output, _ = self.lstm(input)
        # print(f'output:{output.shape}')
        output = self.FC(output[:,-1])
        return output

class Discriminator_LSTM_Gaussian(nn.Module):
    def __init__(self,input_size,hidden_nodes,output_size,batch_size,RNN_depth = 3):
        super().__init__()
        self.hidden_size = hidden_nodes
        self.output_size = output_size
        self.batch_size = batch_size
        self.RNN_depth = RNN_depth
        self.lstm = DPLSTM(input_size = input_size,hidden_size = hidden_nodes ,num_layers = RNN_depth,batch_first= True)
        self.FC = nn.Sequential(
            nn.Linear(in_features = hidden_nodes , out_features = output_size),
            nn.Sigmoid()
        )
    
    def forward(self,input):
        # h0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        # c0 = torch.zeros(self.RNN_depth, self.batch_size, self.hidden_size)
        output, _ = self.lstm(input)
        
        output = self.FC(output[:,-1])
        return output
    

# generator == encoder

def set_models_LSTM(parameters):
    Mech = parameters['Mech']
    generator = LSTM_FC(input_size = 1,hidden_nodes = parameters['hidden_dim'],output_size = parameters['hidden_dim'],batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
    
    if parameters['Mech'] =="Gaussian":
        print(f'discriminator_DPSGD')
        # discriminator = Discriminator_LSTM_Gaussian(input_size = parameters['hidden_dim'],hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
        discriminator = Discriminator_LSTM_Gaussian(input_size = 1,hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
            
    else:
        print(f'Mech:{Mech}')
        discriminator = Discriminator_LSTM(input_size = 1,hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
        # discriminator = Discriminator_LSTM(input_size = parameters['hidden_dim'],hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
    encoder = LSTM_FC(input_size = 1,hidden_nodes =  parameters['hidden_dim'],output_size = parameters['hidden_dim'],batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
    decoder = LSTM_decoder(input_size = parameters['hidden_dim'],hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers'])
    # decoder = Decoder_attention(input_size = parameters['hidden_dim'],hidden_nodes =  parameters['hidden_dim'],output_size = 1,batch_size = parameters['batch_size'])
    supervisor = LSTM_FC(input_size = parameters['hidden_dim'],hidden_nodes = parameters['hidden_dim'],output_size = parameters['hidden_dim'],batch_size = parameters['batch_size'],RNN_depth=parameters['num_layers']-1)
    return encoder,decoder,generator,discriminator,supervisor

def random_generator(batch_size,length):
    return torch.randn(batch_size,length,1).double().to('cuda')
def labels_generator(batch_size,value):
    if value == 1:
        return torch.ones(batch_size,1).double().to('cuda')
    elif value == 0:
        return torch.zeros(batch_size,1).double().to('cuda')
    

def eval_model(models,dataX):
     # 临时测试
    for model in models:
        model.eval()
    for x,label in dataX:
        x = x.to('gpu')
        for model in model:
            x = model(x)
        # h_hat = encoder(x)
        # x_hat = decoder(h_hat) 
        torch.save(x,'./saved_data/p1.data')
        # torch.save(x_hat,'./saved_data/x_hat.data')
        p1 = x 
        # utils.plot_tensor(x,x_hat,'test')
        break
    for x,label in dataX:
        p2 = x
        torch.save(x,'./saved_data/p2.data')
    utils.plot_tensor(p1,p2,'test')

# def train_model(encoder,decoder,generator,discriminator,supervisor,parameters,dataX):
def tgan_pure(dataX,parameters):
    parameters['Mech'] = 'None'
    new_datasets = []
    source_noises = []
    device = ('cuda' if torch.cuda.is_available() else 'cpu' )
    print(f'device: {device}')
    encoder,decoder,generator,discriminator,supervisor = set_models_LSTM(parameters)


    
    # to cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    supervisor = supervisor.to(device)
    # dataX = dataX.to(device)

    # optimizer
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.001)
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr = 0.001)
    optimizer_Su = torch.optim.Adam(supervisor.parameters(),lr = 0.001)

    # H = encoder(dataX)
    # X_reconstructed = decoder(H)

    MSE = nn.MSELoss()
    CE = nn.BCELoss()
    encoder.train()
    encoder.double()
    decoder.train()
    decoder.double()
    print(f'---------------------------------------------------------------------')
    print(f'编码解码模块训练, MSE(En->De) ... ')
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 2 ):
        if is_break :break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            H = encoder(data)
            X_reconstructed = decoder(H)
            optimizer_En.zero_grad()
            optimizer_De.zero_grad()
            L_R = 100 * torch.sqrt( MSE(X_reconstructed,data))
            L_R.backward()
            # encoder decoder 分开迭代
            if  i < int(dataX.__len__()/2):
            # if int(i % 4) <2 :
            # if int(epoch % 2) <  1:
                optimizer_De.step()
            else:
                optimizer_En.step()

            
                
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'H:{H[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H.shape:{H.shape}')
            if L_R.item() < 1.0:
                print(str(datetime.datetime.now()))
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'X_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H:{H[0]}')
                print('编解码训练跳出 ...')
                is_break = True
                break 

    print(f'---------------------------------------------------------------------')
    print(f'生成器加入训练, L_U = MSE(En->U&G->U) ... ')
    generator.train()
    generator.double()
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.0001)
    # 降低En的学习率，微调
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 3):
        if is_break:break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            # 这里只考虑时序特征，判别器不在循环中，因此不输入噪声
            # noise = random_generator(data.shape[0],data.shape[1])
            # noise.to(device)

            H = encoder(data)
            H_hat = generator(data[:,:-1,:])#输入[0：-1）个值，输出[1：-1]，与H[1：-1]做监督损失
            
            x_hat = decoder(H)
            # H_for_decoder = H.to(device)
            # X_hat = decoder(H_for_decoder)
            optimizer_G.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            # print(f'size of H&H_hat:{H.shape},{H_hat.shape}')
            L_U = 100 * torch.sqrt(MSE(H[:,1:,:],H_hat))

            L_R = 100 * torch.sqrt(MSE(data,x_hat))
            LUR = L_U + L_R
            LUR.backward()
            if L_U < 1.0:
                optimizer_En.step()
                optimizer_De.step()
            optimizer_G.step()
            
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                # print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
            if LUR <4.0 :
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                is_break = True
                break 

                

    
    print(f'---------------------------------------------------------------------')
    print(f'判别器加入训练,  ... ')
    discriminator.train()
    discriminator.double()
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.0001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    for epoch in range(parameters['iterations'][-1]):
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data_for_G = data.to(device)
            data = data.to(device)
            
            noise = random_generator(data.shape[0],data.shape[1])
            noise_for_G = noise.to(device)
            # noise.to(device)

            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()


            H_hat = generator(noise)
            H = encoder(data)
            Y = discriminator(decoder( H))
            # Y = CRR.apply(Y_raw,parameters['rd_respons_p'])
            
            
            
            Y_hat = discriminator(decoder(H_hat))
            # X_reconstructed = decoder(H)
            
            # L_U = MSE(H,H_hat)
            
            L_N_fake = CE( Y_hat, labels_generator( data.shape[0], 0) )
            # L_N_fake.backward()
            L_N_real = CE( Y, labels_generator( data.shape[0], 1) )
            # L_N_real.backward()
            L_N = L_N_fake + L_N_real

            # L_R = MSE(X_reconstructed,data)

            # L1 = 100 * torch.sqrt(L_U) + L_N
            # L1.backward()
            # optimizer_G.step()
            # optimizer_En.step()
            # optimizer_D.step()

            # L2 = 100*torch.sqrt(L_U) + 100*torch.sqrt(L_R) + L_N
            L2 = L_N
            L2.backward()
            # if L2.item() > 0.25:#控制判别器强度
                
            # optimizer_En.step()
            optimizer_D.step()
            # optimizer_De.step()
            # 更新生成器
            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()

            H_hat = generator(noise_for_G)
            H = encoder(data_for_G)
                # 为防止重复求导，数据与噪声都与这一轮的数据相同，但是重设一份
            
            Y_hat = discriminator(decoder( H_hat))
            
            # Y = discriminator(H)
            X_reconstructed = decoder(H)
            

            H_U = generator(data_for_G[:,:-1,:])
            
            L_U = 100* torch.sqrt(MSE(H[:,1:,:],H_U))
            L_R = 100*torch.sqrt(MSE(data,X_reconstructed))
            L_N_G = CE(  Y_hat,labels_generator(data.shape[0],1)  ) 

            L3 = L_U +  L_N_G + L_R
            L3.backward()
            optimizer_G.step()  
            optimizer_De.step()  
            optimizer_En.step()   

            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'H :{H[0]}')
                print(f'H_U:{H_U[0]}')
                print(f'data_for_G:{data_for_G[0].view(-1)}')
                print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
                X_new = decoder(H_hat)
                print(f'X_new:{X_new[0].view(-1)}')
                print(f"X_raw:{data_for_G[0].view(-1)}")
                print(f'X_new:{X_new[1].view(-1)}')
                print(f"X_raw:{data_for_G[1].view(-1)}")
                print(f'epoch:{epoch}, 判别器损失 L_N:{L_N.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
                print(f'epoch:{epoch}, 更新生成器的判别器损失 :{L_N_G.item()}, i:{i}')
                print(f'epoch:{epoch}, 隐空间预测监督损失 L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, 重构损失 L_R:{L_R.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器KL损失 L_N_G:{L_N_G.item()}, i:{i}')
                
                

        # if DP.check_privacy(epoch,parameters['iterations']):
        #     generator.eval()
        #     decoder.eval()
        #     print(f'check privacy...')
        #     noise = random_generator(parameters['batch_size'] * dataX.__len__() ,parameters['T'])
        #     H = generator(noise)
        #     new_dataset = decoder(H)
        #     print(f'type of new_dataset:{type(new_dataset)}')
        #     print(f'size of noise:{noise.shape}')
        #     print(f'size of new_dataset:{new_dataset.shape}')

        #     new_datasets.append(new_dataset)
        #     if parameters['white_box']:
        #         source_noises.append(noise)
        #         print(f'white box noises are appended.')
        #     generator.train()
        #     decoder.train()

    generator.eval()
    decoder.eval()
    print(f'check privacy...')
    noise = random_generator(parameters['batch_size'] * dataX.__len__() * parameters['MIA_X'],parameters['T'])
    H = generator(noise)
    new_dataset = decoder(H)
    print(f'type of new_dataset:{type(new_dataset)}')
    print(f'size of noise:{noise.shape}')
    print(f'size of new_dataset:{new_dataset.shape}')

    new_datasets.append(new_dataset)
    if parameters['white_box']:
        source_noises.append(noise)
        print(f'white box noises are appended.')
 
        

    dataname = './newdatas/' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
    new_datasets = torch.tensor( [ x.cpu().detach().numpy() for x in new_datasets] )
    torch.save(new_datasets,dataname)
    if parameters['white_box']:
        noise_name = './newdatas/noise_' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
        source_noises = torch.tensor([ x.cpu().detach().numpy() for x in source_noises])
        torch.save(source_noises,noise_name)
        return new_datasets,source_noises
    else:   
        return new_datasets

def tgan_CRR(dataX,parameters):
    parameters['Mech'] = 'CRR'
    new_datasets = []
    source_noises = []
    device = ('cuda' if torch.cuda.is_available() else 'cpu' )
    print(f'device: {device}')
    encoder,decoder,generator,discriminator,supervisor = set_models_LSTM(parameters)


    
    # to cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    supervisor = supervisor.to(device)
    # dataX = dataX.to(device)

    # optimizer
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.001)
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr = 0.001)
    optimizer_Su = torch.optim.Adam(supervisor.parameters(),lr = 0.001)

    # H = encoder(dataX)
    # X_reconstructed = decoder(H)

    MSE = nn.MSELoss()
    CE = nn.BCELoss()
    encoder.train()
    encoder.double()
    decoder.train()
    decoder.double()
    print(f'---------------------------------------------------------------------')
    print(f'编码解码模块训练, MSE(En->De) ... ')
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 2):
        if is_break :break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            H = encoder(data)
            X_reconstructed = decoder(H)
            optimizer_En.zero_grad()
            optimizer_De.zero_grad()
            L_R = 100* torch.sqrt(MSE(X_reconstructed,data))
            L_R.backward()
            if  i < int(dataX.__len__()/2):
            # if int(i % 4) <2 :
            # if int(epoch % 2) <  1:
                optimizer_De.step()
            else:
                optimizer_En.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                
                print(f'data[0]:{data[0].view(-1)}')
                print(f'X_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H:{H[0]}')
            if L_R.item() < 1.0:
                print(str(datetime.datetime.now()))
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'X_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H:{H[0]}')
                print('编解码训练跳出 ...')
                is_break = True
                break 

    print(f'---------------------------------------------------------------------')
    print(f'生成器加入训练, L_U = MSE(En->U&G->U) ... ')
    generator.train()
    generator.double()
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.0001)
    # 降低En的学习率，微调
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 3):
        if is_break:break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            # 这里只考虑时序特征，判别器不在循环中，因此不输入噪声
            # noise = random_generator(data.shape[0],data.shape[1])
            # noise.to(device)

            H = encoder(data)
            H_hat = generator(data[:,:-1,:])#输入[0：-1）个值，输出[1：-1]，与H[1：-1]做监督损失
            
            x_hat = decoder(H)
            # H_for_decoder = H.to(device)
            # X_hat = decoder(H_for_decoder)
            optimizer_G.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            # print(f'size of H&H_hat:{H.shape},{H_hat.shape}')
            L_U = 100* torch.sqrt(MSE(H[:,1:,:],H_hat))
            L_R = 100* torch.sqrt(MSE(data,x_hat))
            LUR = L_U + L_R
            LUR.backward()
            if L_U < 1.0:
                optimizer_En.step()
                optimizer_De.step()
            
            optimizer_G.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
            if LUR <4.0 :
                print(str(datetime.datetime.now()))
                
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                is_break = True
                break 


    
    print(f'---------------------------------------------------------------------')
    print(f'判别器加入训练,  ... ')
    discriminator.train()
    discriminator.double()
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.0001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    for epoch in range(parameters['iterations'][-1]):
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data_for_G = data.to(device)
            data = data.to(device)
            
            noise = random_generator(data.shape[0],data.shape[1])
            noise_for_G = noise.to(device)
            # noise.to(device)

            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()


            H_hat = generator(noise)
            H = encoder(data)
            Y_raw = discriminator(decoder(H))
            Y = CRR.apply(Y_raw,parameters['rd_respons_p'])
            
            
            Y_hat = discriminator(decoder(H_hat))
            # X_reconstructed = decoder(H)
            # X_reconstructed_hat = decoder(H_hat)
            
            # L_U = MSE(H,H_hat) 
            
            L_N_fake = CE( Y_hat, labels_generator( data.shape[0], 0) )
            # L_N_fake.backward()
            L_N_real = CE( Y, labels_generator( data.shape[0], 1) )
            # L_N_real.backward()
            L_N = L_N_fake + L_N_real

            # L_R = MSE(X_reconstructed,data) 

            # L1 = 100 * torch.sqrt(L_U) + L_N
            # L1.backward()
            # optimizer_G.step()
            # optimizer_En.step()
            # optimizer_D.step()

            # L2 = 100*torch.sqrt(L_U) + 100*torch.sqrt(L_R) + L_N
            L2 = L_N
            L2.backward()
            # if L2.item() > 0.25:#控制判别器强度
                
            # optimizer_En.step()
            optimizer_D.step()
            # optimizer_De.step()
            # 更新生成器
            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()

            H_hat = generator(noise_for_G)
            H = encoder(data_for_G)
                # 为防止重复求导，数据与噪声都与这一轮的数据相同，但是重设一份
            Y_hat = discriminator(decoder(H_hat))
            # Y = discriminator(H)
            X_reconstructed = decoder(H)
            H_U = generator(data_for_G[:,:-1,:])
            
            L_U = 100* torch.sqrt(MSE(H[:,1:,:],H_U))
            
            # L_U = MSE(H,H_hat)
            L_R = 100* torch.sqrt(MSE(data,X_reconstructed))
            L_N_G = CE(  Y_hat,labels_generator(data.shape[0],1)  ) 

            L3 = L_U +  L_N_G + L_R
            L3.backward()
            optimizer_G.step()  
            optimizer_De.step()  
            optimizer_En.step()

            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'H: {H[0]}')
                print(f'H_U:{H_U[0]}')
                print(f'data_for_G:{data_for_G[0].view(-1)}')
                print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
                X_new = decoder(H_hat)
                print(f'X_new:{X_new[0].view(-1)}')
                print(f"X_raw:{data_for_G[0].view(-1)}")
                print(f'X_new:{X_new[1].view(-1)}')
                print(f"X_raw:{data_for_G[1].view(-1)}")
                print(f'epoch:{epoch}, 判别器损失 L_N:{L_N.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
                print(f'epoch:{epoch}, 更新生成器的判别器损失 :{L_N_G.item()}, i:{i}')
                print(f'epoch:{epoch}, 隐空间预测监督损失 L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, 重构损失 L_R:{L_R.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器KL损失 L_N_G:{L_N_G.item()}, i:{i}')
                # print(f'')

        if DP.check_privacy(epoch,parameters['iterations']):
            print(str(datetime.datetime.now()))
            print(f'H: {H[0]}')
            print(f'H_U:{H_U[0]}')
            print(f'data_for_G:{data_for_G[0].view(-1)}')
            print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
            print(f'epoch:{epoch}, 判别器损失 L_N:{L_N.item()}, i:{i}')
            print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
            print(f'epoch:{epoch}, 更新生成器的判别器损失 :{L_N_G.item()}, i:{i}')
            print(f'epoch:{epoch}, 隐空间预测监督损失 L_U:{L_U.item()}, i:{i}')
            print(f'epoch:{epoch}, 重构损失 L_R:{L_R.item()}, i:{i}')
            generator.eval()
            decoder.eval()
            print(f'check privacy...')
            noise = random_generator(parameters['batch_size'] * dataX.__len__() * parameters['MIA_X'] ,parameters['T'])
            H = generator(noise)
            new_dataset = decoder(H)
            print(f'type of new_dataset:{type(new_dataset)}')
            print(f'size of noise:{noise.shape}')
            print(f'size of new_dataset:{new_dataset.shape}')

            new_datasets.append(new_dataset)
            if parameters['white_box']:
                source_noises.append(noise)
                print(f'white box noises are appended.')
            generator.train()
            decoder.train()
        

    dataname = './newdatas/' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
    new_datasets = torch.tensor( [ x.cpu().detach().numpy() for x in new_datasets] )
    torch.save(new_datasets,dataname)
    if parameters['white_box']:
        noise_name = './newdatas/noise_' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
        source_noises = torch.tensor([ x.cpu().detach().numpy() for x in source_noises])
        torch.save(source_noises,noise_name)
        return new_datasets,source_noises
    else:   
        return new_datasets


def tgan_DPSGD(dataX,parameters):
    parameters['Mech'] = 'Gaussian'
    count_epsilons = []
    print(f'DPSGD ...')
    new_datasets = []
    source_noises = []
    device = ('cuda' if torch.cuda.is_available() else 'cpu' )
    print(f'device: {device}')
    encoder,decoder,generator,discriminator,supervisor = set_models_LSTM(parameters)


    
    # to cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    supervisor = supervisor.to(device)
    # dataX = dataX.to(device)

    # optimizer
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.001)
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    optimizer_D = torch.optim.SGD(discriminator.parameters(),lr = 0.001)
    optimizer_Su = torch.optim.Adam(supervisor.parameters(),lr = 0.001)

    # H = encoder(dataX)
    # X_reconstructed = decoder(H)

    MSE = nn.MSELoss()
    CE = nn.BCELoss()
    encoder.train()
    encoder.double()
    decoder.train()
    decoder.double()
    print(f'---------------------------------------------------------------------')
    print(f'编码解码模块训练, MSE(En->De) ... ')
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 2):
        if is_break :break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            H = encoder(data)
            X_reconstructed = decoder(H)
            optimizer_En.zero_grad()
            optimizer_De.zero_grad()
            L_R = 100* torch.sqrt(MSE(X_reconstructed,data))
            L_R.backward()
            if  i < int(dataX.__len__()/2):
            # if int(i % 4) <2 :
            # if int(epoch % 2) <  1:
                optimizer_De.step()
            else:
                optimizer_En.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'X_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H:{H[0]}')
            if L_R.item() < 1.0:
                print(str(datetime.datetime.now()))
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'X_reconstructed:{X_reconstructed[0].view(-1)}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                print(f'H:{H[0]}')
                print('编解码训练跳出 ...')
                is_break = True
                break 


    print(f'---------------------------------------------------------------------')
    print(f'生成器加入训练, L_U = MSE(En->U&G->U) ... ')
    generator.train()
    generator.double()
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.0001)
    # 降低En的学习率，微调
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 3):
        if is_break:break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            # 这里只考虑时序特征，判别器不在循环中，因此不输入噪声
            # noise = random_generator(data.shape[0],data.shape[1])
            # noise.to(device)

            H = encoder(data)
            H_hat = generator(data[:,:-1,:])#输入[0：-1）个值，输出[1：-1]，与H[1：-1]做监督损失
            
            x_hat = decoder(H)
            # H_for_decoder = H.to(device)
            # X_hat = decoder(H_for_decoder)
            optimizer_G.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            # print(f'size of H&H_hat:{H.shape},{H_hat.shape}')
            L_U = 100* torch.sqrt(MSE(H[:,1:,:],H_hat))
            L_R = 100* torch.sqrt(MSE(data,x_hat))
            LUR = L_U + L_R
            LUR.backward()
            if L_U < 1.0:
                optimizer_En.step()
                optimizer_De.step()
            
            optimizer_G.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
            if LUR <4.0 :
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                is_break = True
                break 


    
    print(f'---------------------------------------------------------------------')
    print(f'判别器加入训练,  ... ')
    discriminator.train()
    discriminator.double()
    # DPSGD _____________________________________________________
    privacy_engine = PrivacyEngine()
    discriminator,optimizer_D,dataX = privacy_engine.make_private(
        module = discriminator,
        optimizer=optimizer_D,
        data_loader=dataX,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )
    # privacy_engine.attach(optimizer_D)
    # ___________________________________________________________________
    
    discriminator.train()
    discriminator.double()
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.0001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)

    
    epsilon_0 = 0
    for epoch in range(parameters['iterations'][-1]):
    # for epoch in range(30):
        for i ,(data,label) in enumerate(dataX):
            data_for_G = data.to(device)
            data = data.to(device)
            
            noise = random_generator(data.shape[0],data.shape[1])
            noise_for_G = noise.to(device)
            # noise.to(device)

            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()


            H_hat = generator(noise)
            H = encoder(data)
            Y = discriminator(decoder(H))
            # Y = CRR.apply(Y_raw,parameters['rd_respons_p'])
            
            
            Y_hat = discriminator(decoder(H_hat))
            # X_reconstructed = decoder(H)
            
            # L_U = MSE(H,H_hat)
            
            L_N_fake = CE( Y_hat, labels_generator( data.shape[0], 0) )
            # L_N_fake.backward()
            L_N_real = CE( Y, labels_generator( data.shape[0], 1) )
            # L_N_real.backward()
            L_N = L_N_fake + L_N_real

            # L_R = MSE(X_reconstructed,data)

            # L1 = 100 * torch.sqrt(L_U) + L_N
            # L1.backward()
            # optimizer_G.step()
            # optimizer_En.step()
            # optimizer_D.step()

            # L2 = 100*torch.sqrt(L_U) + 100*torch.sqrt(L_R) + L_N
            L2 = L_N
            L2.backward()
            # if L2.item() > 0.25:#控制判别器强度
                
            # optimizer_En.step()
            optimizer_D.step()
            # optimizer_De.step()
            # 更新生成器
            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()

            H_hat = generator(noise_for_G)
            H = encoder(data_for_G)
                # 为防止重复求导，数据与噪声都与这一轮的数据相同，但是重设一份
            Y_hat = discriminator(decoder(H_hat))
            # Y = discriminator(H)
            X_reconstructed = decoder(H)
            H_U = generator(data_for_G[:,:-1,:])
            
            L_U = 100* torch.sqrt(MSE(H[:,1:,:],H_U))
            
            # L_U = MSE(H,H_hat)
            L_R = 100* torch.sqrt(MSE(data,X_reconstructed))
            L_N_G = CE(  Y_hat,labels_generator(data.shape[0],1)  ) 

            L3 = L_U + L_N_G + L_R
            L3.backward()
            optimizer_G.step()  
            optimizer_De.step()  
            optimizer_En.step()  
            # ________________________________________________________
            epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta = 1e-5)
            
            # __________________________________________________________
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'H: {H[0]}')
                print(f'H_U:{H_U[0]}')
                print(f'data_for_G:{data_for_G[0].view(-1)}')
                print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
                X_new = decoder(H_hat)
                print(f'X_new:{X_new[0].view(-1)}')
                print(f"X_raw:{data_for_G[0].view(-1)}")
                print(f'X_new:{X_new[1].view(-1)}')
                print(f"X_raw:{data_for_G[1].view(-1)}")
                print(f'epoch:{epoch}, 判别器损失 L_N:{L_N.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
                print(f'epoch:{epoch}, 更新生成器的判别器损失 :{L_N_G.item()}, i:{i}')
                print(f'epoch:{epoch}, 隐空间预测监督损失 L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, 重构损失 L_R:{L_R.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器KL损失 L_N_G:{L_N_G.item()}, i:{i}')
                print(f'epsilon:{epsilon},best_alpha:{best_alpha}')

        # if DP.check_privacy(epoch,parameters['iterations']):
        if DP.check_privacy_DPSGD(epsilon,epsilon_0,parameters['epsilons']):
            print(str(datetime.datetime.now()))
            print(f'H: {H[0]}')
            print(f'H_U:{H_U[0]}')
            print(f'data_for_G:{data_for_G[0].view(-1)}')
            print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
            print(f'epoch:{epoch}, 判别器损失 L_N:{L_N.item()}, i:{i}')
            print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
            print(f'epoch:{epoch}, 更新生成器的判别器损失 :{L_N_G.item()}, i:{i}')
            print(f'epoch:{epoch}, 隐空间预测监督损失 L_U:{L_U.item()}, i:{i}')
            print(f'epoch:{epoch}, 重构损失 L_R:{L_R.item()}, i:{i}')
            print(f'epoch:{epoch}, 生成器KL损失 L_N_G:{L_N_G.item()}, i:{i}')
            print(f'epsilon:{epsilon},best_alpha:{best_alpha}')
            count_epsilons.append(round(epsilon,2))
            generator.eval()
            decoder.eval()
            print(f'check privacy for DPSGD...')
            noise = random_generator(parameters['batch_size'] * dataX.__len__() * parameters['MIA_X'],parameters['T'])
            H = generator(noise)
            new_dataset = decoder(H)
            print(f'type of new_dataset:{type(new_dataset)}')
            print(f'size of noise:{noise.shape}')
            print(f'size of new_dataset:{new_dataset.shape}')

            new_datasets.append(new_dataset)
            if parameters['white_box']:
                source_noises.append(noise)
                print(f'white box noises are appended.')
            generator.train()
            decoder.train()
        if epsilon >= 9.0:break
        epsilon_0 = epsilon
        # if epsilon_0 == -1:
        #         epsilon_0 = epsilon
        # else:
        #     epsilon_0 = epsilon - epsilon_0

    parameters['epsilons'] = count_epsilons 

    dataname = './newdatas/' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
    new_datasets = torch.tensor( [ x.cpu().detach().numpy() for x in new_datasets] )
    torch.save(new_datasets,dataname)
    if parameters['white_box']:
        noise_name = './newdatas/noise_' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
        source_noises = torch.tensor([ x.cpu().detach().numpy() for x in source_noises])
        torch.save(source_noises,noise_name)
        return new_datasets,source_noises
    else:   
        return new_datasets  





    # 临时测试
    # encoder.eval()
    # decoder.eval()
    # for x,label in dataX:
    #     x = x.to(device)
    #     h_hat = encoder(x)
    #     x_hat = decoder(h_hat) 
    #     torch.save(x,'./saved_data/x.data')
    #     torch.save(x_hat,'./saved_data/x_hat.data')
    #     utils.plot_tensor(x,x_hat,'test')
    #     break
# def train_model(encoder,decoder,generator,discriminator,supervisor,parameters,dataX):
def tgan_Gsu(dataX,parameters):
    parameters['Mech'] = 'None'
    new_datasets = []
    source_noises = []
    device = ('cuda' if torch.cuda.is_available() else 'cpu' )
    print(f'device: {device}')
    encoder,decoder,generator,discriminator,supervisor = set_models_LSTM(parameters)


    
    # to cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    supervisor = supervisor.to(device)
    # dataX = dataX.to(device)

    # optimizer
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.001)
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(),lr = 0.001)
    optimizer_Su = torch.optim.Adam(supervisor.parameters(),lr = 0.001)

    # H = encoder(dataX)
    # X_reconstructed = decoder(H)

    MSE = nn.MSELoss()
    CE = nn.BCELoss()
    encoder.train()
    encoder.double()
    decoder.train()
    decoder.double()
    print(f'---------------------------------------------------------------------')
    print(f'编码解码模块训练, MSE(En->De) ... ')
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 2):
        if is_break :break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            H = encoder(data)
            X_reconstructed = decoder(H)
            optimizer_En.zero_grad()
            optimizer_De.zero_grad()
            L_R = 100* torch.sqrt(MSE(X_reconstructed,data))
            L_R.backward()
            if  i < int(dataX.__len__()/2):
            # if int(i % 4) <2 :
            # if int(epoch % 2) <  1:
                optimizer_De.step()
            else:
                optimizer_En.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')

    print(f'---------------------------------------------------------------------')
    print(f'生成器加入训练, L_U = MSE(En->U&G->U) ... ')
    generator.train()
    generator.double()
    optimizer_En = torch.optim.Adam(encoder.parameters(),lr = 0.0001)
    # 降低En的学习率，微调
    is_break = False
    for epoch in range(parameters['iterations'][-1] * 3):
        if is_break:break
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data = data.to(device)
            # 这里只考虑时序特征，判别器不在循环中，因此不输入噪声
            # noise = random_generator(data.shape[0],data.shape[1])
            # noise.to(device)

            H = encoder(data)
            H_hat = generator(data[:,:-1,:])#输入[0：-1）个值，输出[1：-1]，与H[1：-1]做监督损失
            
            x_hat = decoder(H)
            # H_for_decoder = H.to(device)
            # X_hat = decoder(H_for_decoder)
            optimizer_G.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            # print(f'size of H&H_hat:{H.shape},{H_hat.shape}')
            L_U = 100* torch.sqrt(MSE(H_hat,H[:,1:,:]))

            L_R = 100* torch.sqrt(MSE(x_hat,data))
            LUR = L_U + L_R
            LUR.backward()
            if L_U < 1.0:
                optimizer_En.step()
                optimizer_De.step()
            optimizer_G.step()
            if epoch%50 == 0 and i == 0:
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                
            if LUR <4.0 :
                print(str(datetime.datetime.now()))
                print(f'ecoded H:{H[0]}')
                print(f'H_hat:{H_hat[0]}')
                print(f'data[0]:{data[0].view(-1)}')
                print(f'x_hat :{x_hat[0].view(-1)}')
                print(f'epoch:{epoch}, L_U:{L_U.item()}, i:{i}')
                print(f'epoch:{epoch}, L_R:{L_R.item()}, i:{i}')
                is_break = True
                break 


    
    print(f'---------------------------------------------------------------------')
    print(f'判别器加入训练,  ... ')
    discriminator.train()
    discriminator.double()
    optimizer_De = torch.optim.Adam(decoder.parameters(),lr = 0.0001)
    optimizer_G = torch.optim.Adam(generator.parameters(),lr = 0.001)
    for epoch in range(parameters['iterations'][-1]):
    # for epoch in range(5):
        for i ,(data,label) in enumerate(dataX):
            data_for_G = data.to(device)
            data = data.to(device)
            
            noise = random_generator(data.shape[0],data.shape[1])
            noise_for_G = noise.to(device)
            # noise.to(device)

            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()


            H_hat = generator(noise)
            H = encoder(data)
            Y = discriminator(decoder(H))
            # Y = CRR.apply(Y_raw,parameters['rd_respons_p'])
            
            
            
            Y_hat = discriminator(decoder(H_hat))
            # X_reconstructed = decoder(H)
            
            # L_U = MSE(H,H_hat)
            
            L_N_fake = CE( Y_hat, labels_generator( data.shape[0], 0) )
            # L_N_fake.backward()
            L_N_real = CE( Y, labels_generator( data.shape[0], 1) )
            # L_N_real.backward()
            L_N = L_N_fake + L_N_real

            # L_R = MSE(X_reconstructed,data)

            # L1 = 100 * torch.sqrt(L_U) + L_N
            # L1.backward()
            # optimizer_G.step()
            # optimizer_En.step()
            # optimizer_D.step()

            # L2 = 100*torch.sqrt(L_U) + 100*torch.sqrt(L_R) + L_N
            L2 = L_N
            L2.backward()
            # if L2.item() > 0.25:#控制判别器强度
               
            # optimizer_En.step()
            optimizer_D.step()
            # optimizer_De.step()
            # 更新生成器
            optimizer_D.zero_grad()
            optimizer_De.zero_grad()
            optimizer_En.zero_grad()
            optimizer_G.zero_grad()

            H_hat = generator(noise_for_G)
            H = encoder(data_for_G)
                # 为防止重复求导，数据与噪声都与这一轮的数据相同，但是重设一份
            Y_hat = discriminator(decoder(H_hat))
            # Y = discriminator(H)
            X_reconstructed = decoder(H)

            H_U = generator(data_for_G[:,:-1,:])



            
            L_U = 100* torch.sqrt(MSE(H_U,H[:,1:,:]))
            L_R = 100* torch.sqrt(MSE(X_reconstructed,data))
            L_N_G = CE(  Y_hat,labels_generator(data.shape[0],1)  ) 

            L3 = L_U +  L_N_G + L_R
            L3.backward()
            optimizer_G.step()  
            optimizer_De.step()  
            optimizer_En.step()   

            if epoch%50 == 0 and i == 0:
                
                print(str(datetime.datetime.now()))
                print(f'H: {H[0]}')
                print(f'H_U:{H_U[0]}')
                print(f'data_for_G:{data_for_G[0].view(-1)}')
                print(f'x_reconstruct:{X_reconstructed[0].view(-1)}')
                X_new = decoder(H_hat)
                print(f'X_new:{X_new[0].view(-1)}')
                print(f"X_raw:{data_for_G[0].view(-1)}")
                print(f'X_new:{X_new[1].view(-1)}')
                print(f"X_raw:{data_for_G[1].view(-1)}")
                print(f'epoch:{epoch}, 判别器混合损失 L_2:{L2.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器混合损失 L_3:{L3.item()}, i:{i}')
                print(f'epoch:{epoch}, 生成器KL损失 L_N_G:{L_N_G.item()}, i:{i}')
               

        # if DP.check_privacy(epoch,parameters['iterations']):
        #     generator.eval()
        #     decoder.eval()
        #     print(f'check privacy...')
        #     noise = random_generator(parameters['batch_size'] * dataX.__len__() ,parameters['T'])
        #     H = generator(noise)
        #     new_dataset = decoder(H)
        #     print(f'type of new_dataset:{type(new_dataset)}')
        #     print(f'size of noise:{noise.shape}')
        #     print(f'size of new_dataset:{new_dataset.shape}')

        #     new_datasets.append(new_dataset)
        #     if parameters['white_box']:
        #         source_noises.append(noise)
        #         print(f'white box noises are appended.')
        #     generator.train()
        #     decoder.train()

    generator.eval()
    decoder.eval()
    print(f'check privacy...')
    noise = random_generator(parameters['batch_size'] * dataX.__len__() * parameters['MIA_X'] ,parameters['T'])
    H = generator(noise)
    new_dataset = decoder(H)
    print(f'type of new_dataset:{type(new_dataset)}')
    print(f'size of noise:{noise.shape}')
    print(f'size of new_dataset:{new_dataset.shape}')

    new_datasets.append(new_dataset)
    if parameters['white_box']:
        source_noises.append(noise)
        print(f'white box noises are appended.')
 
        

    dataname = './newdatas/' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
    new_datasets = torch.tensor( [ x.cpu().detach().numpy() for x in new_datasets] )
    torch.save(new_datasets,dataname)
    if parameters['white_box']:
        noise_name = './newdatas/noise_' + parameters['dataset'] + '_' + parameters['Mech'] + '.data'
        source_noises = torch.tensor([ x.cpu().detach().numpy() for x in source_noises])
        torch.save(source_noises,noise_name)
        return new_datasets,source_noises
    else:   
        return new_datasets






 

    

