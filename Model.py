import torch
import torch.nn as nn
import numpy as np
from bmdet import BayesianMDeT
from utils.losses import rmse_loss
from utils.taskbalance import taskbalance
###########################################
class MyModel(nn.Module):
    def __init__(self,
                 cuda=True):
        super(MyModel, self).__init__()
        self.create()
        self.cuda = cuda
    ###########################

    def create(self):
        #self.model = torch.compile(BayesianMDeT(ahead = 1))
        self.model = BayesianMDeT(ahead = 1)

        self.BayesianWeightLinear = taskbalance()

        if self.cuda:
            self.model.cuda()
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.BayesianWeightLinear.parameters()}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ###########################
    
    def fit(self, x, y, samples=1):
        self.optimizer.zero_grad()
        ave_loss1, ave_loss2, ave_loss3, kl = self.model.sample_elbo_m(inputs=x,
                                      labels=y,
                                      sample_nbr=samples)

        overall_loss = self.BayesianWeightLinear(ave_loss1,ave_loss2,ave_loss3)

        overall_loss = overall_loss + kl

        p_mu = self.BayesianWeightLinear.weight_mu

        p_rho = self.BayesianWeightLinear.weight_rho

        overall_loss.backward()
        self.optimizer.step()

        return overall_loss, ave_loss1, ave_loss2, ave_loss3, p_mu, p_rho
    ###########################
    ###########################
    def Mytest(self,
             x_test,
             samples = 10,
             ahead = 1):

        batch_size = x_test.shape[0]

        output1_a = np.zeros((0,batch_size,ahead))
        output2_a = np.zeros((0,batch_size,ahead))
        output3_a = np.zeros((0,batch_size,ahead))

        for i in range(samples):
            output1,output2,output3 = self.model(x_test)

            output1 = np.reshape(output1.cpu().detach().numpy(),((-1,output1.shape[0],output1.shape[1])))
            output2 = np.reshape(output2.cpu().detach().numpy(),((-1,output2.shape[0],output2.shape[1])))
            output3 = np.reshape(output3.cpu().detach().numpy(),((-1,output3.shape[0],output3.shape[1])))

            output1_a = np.concatenate([output1_a,output1], axis = 0)
            output2_a = np.concatenate([output2_a,output2], axis = 0)
            output3_a = np.concatenate([output3_a,output3], axis = 0)

        return output1_a, output2_a, output3_a
    ###########################

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())
#############


