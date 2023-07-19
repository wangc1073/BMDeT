import torch
import numpy as np
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score #R2
#######################################################
def mape_loss(pre,true):
    dataabs = torch.abs(true - pre)
    return torch.mean(dataabs / true)
#######################################################
def mae_loss(pre,true):
    return torch.mean(torch.abs(true - pre))
#######################################################

#######################################################
def rmse_loss(true,pre):
    # RMSE = np.sqrt(np.mean(np.square(true - pre))) 
    # .pow(2)
    #RMSE = (pre - true).norm(2)
    criterion=torch.nn.MSELoss(reduction="mean")
    RMSELoss=torch.sqrt(criterion(pre,true))
    return RMSELoss
#######################################################
def pinball_loss(q,labels,out):
    # labels [batch,24]
    labels = labels.reshape([-1, 1])
    out = out.reshape([-1, 1])
    lossout = 0
    for count in range(labels.shape[0]):
        if labels[count] > out[count]:
            lossout = lossout + q*(labels[count] - out[count])
        else:
            lossout = lossout + (1-q)*(out[count] - labels[count])
    return lossout/out.shape[0]
############################
def crps_func(labels,out):
    # labels: ( size, 1 )
    # out: ( Nsamples, size, 1 )
    Nsamples = out.shape[0]
    Nsize = out.shape[1]

    labels = labels.reshape([Nsize])
    out = out.reshape([-1, Nsize]).T
    # labels: ( size,)
    # out: ( size, Nsamples)
    crpss = 0
    for siz in range(Nsize):
        thelabel = labels[siz]
        theout = out[siz,:]

        crps, fcrps, acrps = crps_func_base(theout,thelabel,Nsamples)

        crpss += crps
    crpss = crpss / Nsize
    return crpss
############################

def rmse_func(true,pre):
    #RMSE = np.sqrt(np.mean(np.square(true - pre))) 
    return np.sqrt(mean_squared_error(true,pre))
#######################################################

def crps_func_base(ensemble_members,observation,adjusted_ensemble_size=200):
    fc = np.sort(ensemble_members)
    ob = observation
    _m = len(fc)
    M = int(adjusted_ensemble_size)
    _cdf_fc = None
    _cdf_ob = None
    _delta_fc = None
    crps = None
    fcrps = None
    acrps = None

    if (ob is not np.nan) and (not np.isnan(fc).any()):
        if ob < fc[0]:
            _cdf_fc = np.linspace(0,(_m - 1)/_m,_m)
            _cdf_ob = np.ones(_m)
            all_mem = np.array([ob] + list(fc), dtype = object)
            _delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)], dtype=object)
            
        elif ob > fc[-1]:
            _cdf_fc = np.linspace(1/_m,1,_m)
            _cdf_ob = np.zeros(_m)
            all_mem = np.array(list(fc) + [ob], dtype = object)
            _delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)], dtype=object) 

        elif ob in fc:
            _cdf_fc = np.linspace(1/_m,1,_m)
            _cdf_ob = (fc >= ob)
            all_mem = fc
            _delta_fc = np.array([all_mem[n+1] - all_mem[n] for n in range(len(all_mem)-1)] + list(np.zeros(1)), dtype=object) 

        else:
            cdf_fc = []
            cdf_ob = []
            delta_fc = []
            for f in range(len(fc)-1):
                if (fc[f] < ob) and (fc[f+1] < ob):
                    cdf_fc.append((f+1)*1/_m)
                    cdf_ob.append(0)
                    delta_fc.append(fc[f+1] - fc[f])
                elif (fc[f] < ob) and (fc[f+1] > ob):
                    cdf_fc.append((f+1)*1/_m)
                    cdf_fc.append((f+1)*1/_m)
                    cdf_ob.append(0)
                    cdf_ob.append(1)
                    delta_fc.append(ob - fc[f])
                    delta_fc.append(fc[f+1] - ob)
                else:
                    cdf_fc.append((f+1)*1/_m)
                    cdf_ob.append(1)
                    delta_fc.append(fc[f+1] - fc[f])
            _cdf_fc = np.array(cdf_fc)
            _cdf_ob = np.array(cdf_ob)
            _delta_fc = np.array(delta_fc)
        
        crps = np.sum(np.array((_cdf_fc - _cdf_ob) ** 2)*_delta_fc)
        if _m == 1:
            fcrps = acrps = 'Not defined'
        else:
            fcrps = crps - np.sum(np.array(((_cdf_fc * (1 - _cdf_fc))/(_m-1))*_delta_fc))
            acrps = crps - np.sum(np.array((((1 - (_m/M)) * _cdf_fc * (1 - _cdf_fc))/(_m-1))*_delta_fc))
        return crps, fcrps, acrps
    else:
        return np.nan, np.nan, np.nan