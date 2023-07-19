import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch
import datetime
import os

from Model import MyModel

from torch.utils.tensorboard import SummaryWriter

# You need to customize the ReadData function here
from readData import getDataforTrainVal

# GPU?
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#############
x_train, y_train = loadDataforTrainVal(input_size=24 * 3, output_size=1)
####################
batch_size = 64
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
####################
writer = SummaryWriter('logfiles')
####################
savepath = 'modelsave/bmdet/'
if not (os.path.exists(savepath)):
    os.makedirs(savepath)
####################
net = MyModel()
print('    Total params: %.2fM' % (np.sum(p.numel() for p in net.parameters()) / 1000000.0))
####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("using GPU")
else:
    print("using CPU")
net.to(device)
Nbatch = len(train_loader)
####################
totolepoch = 100
loss_train = np.zeros((totolepoch, 3))
####################
mu_list = np.zeros((totolepoch, 3))
rho_list = np.zeros((totolepoch, 3))
for epoch in range(totolepoch):
    ####################
    if epoch == 0:
        ELBO_samples = 5
    else:
        ELBO_samples = 1
    ####################
    nb_samples = 0
    lastloss = 0
    ####################

    for i, data in enumerate(train_loader):
        start_time = datetime.datetime.now()

        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()

        overall_loss, loss1, loss2, loss3, p_mu, p_rho = net.fit(x=inputs, y=labels, samples=ELBO_samples)

        mu_list[epoch] += p_mu.cpu().detach().numpy().reshape(3, )
        rho_list[epoch] += p_rho.cpu().detach().numpy().reshape(3, )
        print(p_mu.cpu().detach().numpy().reshape(3, ))
        print(p_rho.cpu().detach().numpy().reshape(3, ))

        loss_train[epoch,0] += loss1
        loss_train[epoch,1] += loss2
        loss_train[epoch,2] += loss3
        nb_samples += len(inputs)

        end_time = datetime.datetime.now()
        lasttime = (end_time - start_time) * (Nbatch - i) + (end_time - start_time) * Nbatch * (
                    totolepoch - epoch - 1)
        print(" eta: ", lasttime,
                " epoch: %4d in %4d, batch: %5d  loss: %.4f LossChange: %.4f  loss1: %.4f  loss2: %.4f  loss3: %.4f " % (
                epoch + 1, totolepoch, (i + 1), overall_loss.item(), overall_loss.item() - lastloss, loss1.item(),
                loss2.item(), loss3.item()))
        ####################
        lastloss = overall_loss.item()
        ####################
        writer.add_scalar('loss', overall_loss.item(), Nbatch*epoch + i )
        writer.add_scalar('loss1', loss1.item(), Nbatch*epoch + i )
        writer.add_scalar('loss2', loss2.item(), Nbatch*epoch + i )
        writer.add_scalar('loss3', loss3.item(), Nbatch*epoch + i )

        writer.add_scalar('p_mu0', p_mu.cpu().detach().numpy()[0].item(), Nbatch*epoch + i )
        writer.add_scalar('p_rho0', p_rho.cpu().detach().numpy()[0].item(), Nbatch*epoch + i )

        writer.add_scalar('p_mu1', p_mu.cpu().detach().numpy()[1].item(), Nbatch*epoch + i )
        writer.add_scalar('p_rho1', p_rho.cpu().detach().numpy()[1].item(), Nbatch*epoch + i )

        writer.add_scalar('p_mu2', p_mu.cpu().detach().numpy()[2].item(), Nbatch*epoch + i )
        writer.add_scalar('p_rho2', p_rho.cpu().detach().numpy()[2].item(), Nbatch*epoch + i )

    loss_train[epoch] = loss_train[epoch] / Nbatch
    mu_list[epoch] = mu_list[epoch] / Nbatch
    rho_list[epoch] = rho_list[epoch] / Nbatch

torch.save(net, savepath + 'bmdet.pt')

np.save(savepath + "mu_list", np.asarray(mu_list))
np.save(savepath + "rho_list", np.asarray(rho_list))

np.save(savepath + "loss_train", np.asarray(loss_train))

print('Finished Training')
####################
