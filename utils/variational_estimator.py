from utils.losses import rmse_loss
from modules.linear_bayesian_layer import BayesianLinear

def variational_estimator(nn_class):
    def nn_kl_divergence(self):
        kl_divergence = 0
        for module in self.modules():
            if isinstance(module, (BayesianLinear)):
                kl_divergence += module.log_variational_posterior - module.log_prior
        return kl_divergence
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo_m(self, inputs, labels, sample_nbr):
        loss1 = 0
        loss2 = 0
        loss3 = 0
        for _ in range(sample_nbr):
            output1, output2, output3 = self(inputs)
            label1 = labels[ : , : , 0 ]
            label2 = labels[ : , : , 1 ]
            label3 = labels[ : , : , 2 ]
            loss1 += rmse_loss(output1, label1)
            loss2 += rmse_loss(output2, label2)
            loss3 += rmse_loss(output3, label3)
        kl= self.nn_kl_divergence()
        kl = kl * 1e-5
        return loss1 / sample_nbr, loss2 / sample_nbr, loss3 / sample_nbr, kl
    setattr(nn_class, "sample_elbo_m", sample_elbo_m)

    return nn_class
