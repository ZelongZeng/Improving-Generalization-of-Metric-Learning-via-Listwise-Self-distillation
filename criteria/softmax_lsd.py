import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### This Implementation follows: https://github.com/azgo14/classification_metric_learning

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.par         = opt

        self.temperature = opt.loss_softmax_temperature

        self.class_map = torch.nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.name           = 'softmax'

        self.lr = opt.loss_softmax_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ###
        self.f_distillation_standard = opt.f_distillation_standard
        self.tau = opt.kd_tau
        self.alpha = opt.kd_alpha
        self.beta = opt.kd_beta
        self.n_epochs = opt.n_epochs

    def logsoftmax(self, x, tau):
        ls = torch.nn.LogSoftmax(dim=1)
        return ls(x / tau)

    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x / tau)

    def kl_d(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(input, target)


    def forward(self, batch, teacher_batch, labels, epoch, **kwargs):
        class_mapped_batch = torch.nn.functional.linear(batch, torch.nn.functional.normalize(self.class_map, dim=1))

        loss_rank = torch.nn.CrossEntropyLoss()(class_mapped_batch/self.temperature, labels.to(torch.long).to(self.par.device))

        # lsd
        similarity = batch.mm(batch.T)
        teacher_similarity = torch.mm(teacher_batch, teacher_batch.t())

        # loss_rank = []
        loss_kd = []
        for i in range(len(batch)):

            # ----------- KD --------------------------
            pos_idxs_kd = labels == labels[i]
            # pos_idxs_kd[i] = 0
            neg_idxs_kd = labels != labels[i]

            if self.f_distillation_standard:
                pos_sim_kd = similarity[i][pos_idxs_kd].unsqueeze(dim=0)
                neg_sim_kd = similarity[i][neg_idxs_kd].unsqueeze(dim=0)

                teacher_pos_sim_kd = teacher_similarity[i][pos_idxs_kd].unsqueeze(dim=0)
                teacher_neg_sim_kd = teacher_similarity[i][neg_idxs_kd].unsqueeze(dim=0)
            else:
                pos_sim_kd = similarity[i][pos_idxs_kd].unsqueeze(dim=0)
                neg_sim_kd = self.beta * similarity[i][neg_idxs_kd].unsqueeze(dim=0)

                teacher_pos_sim_kd = teacher_similarity[i][pos_idxs_kd].unsqueeze(dim=0)
                teacher_neg_sim_kd = self.beta * teacher_similarity[i][neg_idxs_kd].unsqueeze(dim=0)

            sim_cache = self.logsoftmax(torch.cat((pos_sim_kd, neg_sim_kd), dim=1), self.tau)
            target_cache = self.softmax(torch.cat((teacher_pos_sim_kd, teacher_neg_sim_kd), dim=1), self.tau)

            loss_kd.append(self.kl_d(sim_cache, target_cache.detach()))

        # loss_rank = torch.mean(torch.stack(loss_rank))
        loss_kd = torch.mean(torch.stack(loss_kd))

        loss = torch.mean(loss_rank) + (epoch / self.n_epochs) * self.alpha * (self.tau ** 2) * torch.mean(loss_kd)

        return loss, torch.mean(loss_rank), torch.mean(loss_kd)

