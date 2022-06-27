import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.margin             = opt.loss_margin_margin
        self.nu                 = opt.loss_margin_nu
        self.beta_constant      = opt.loss_margin_beta_constant
        self.beta_val           = opt.loss_margin_beta

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_margin_beta)

        self.batchminer = batchminer

        self.name  = 'margin'

        self.lr    = opt.loss_margin_beta_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ###
        self.f_distillation_standard = opt.f_distillation_standard
        self.tau = opt.kd_tau
        self.alpha = opt.kd_alpha
        # self.beta = opt.kd_beta
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
        sampled_triplets = self.batchminer(batch, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss_rank = torch.sum(pos_loss+neg_loss)
            else:
                loss_rank = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: 
                beta_regularization_loss = torch.sum(beta)
                loss_rank += self.nu * beta_regularisation_loss.to(torch.float).to(d_ap.device)
        else:
            loss_rank = torch.tensor(0.).to(torch.float).to(batch.device)

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
