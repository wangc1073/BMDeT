import torch.nn as nn
import torch
####################
class taskbalance(nn.Module):
    def __init__(self,num=3):
        super(taskbalance, self).__init__()

        w_mu = torch.Tensor([0.5, 0.5, 0.5])
        w_rho = torch.Tensor([0.01, 0.01, 0.01])

        self.weight_mu = nn.Parameter(w_mu)
        self.weight_rho = nn.Parameter(w_rho)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, loss1, loss2, loss3):

        std1 = torch.randn([1]).to(self.device)
        std2 = torch.randn([1]).to(self.device)
        std3 = torch.randn([1]).to(self.device)

        w0 = (std1*self.weight_rho[0] + self.weight_mu[0])
        w1 = (std2*self.weight_rho[1] + self.weight_mu[1])
        w2 = (std3*self.weight_rho[2] + self.weight_mu[2])

        loss = 3

        loss = loss + 0.5 / (w0 ** 4) * loss1 + torch.log(w0 ** 2)
        loss = loss + 0.5 / (w1 ** 4) * loss2 + torch.log(w1 ** 2)
        loss = loss + 0.5 / (w2 ** 4) * loss3 + torch.log(w2 ** 2)

        #loss = loss + torch.exp(-w0) * loss1 + torch.exp(w0)
        #loss = loss + torch.exp(-w1) * loss2 + torch.exp(w1)
        #loss = loss + torch.exp(-w2) * loss3 + torch.exp(w2)

        print("")
        print("weight_mu",self.weight_mu.data)
        print("weight_rho",self.weight_rho.data)
        print("sample sigma1:",w0**2)
        print("sample sigma2:",w1**2)
        print("sample sigma3:",w2**2)
        print("")

        return loss
#################
