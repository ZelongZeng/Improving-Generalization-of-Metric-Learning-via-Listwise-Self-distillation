import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.batchminer = batchminer

        self.name           = 'contrastive'

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
        sampled_triplets = self.batchminer(batch, labels)

        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        pos_dists = torch.mean(F.relu(nn.PairwiseDistance(p=2)(batch[anchors,:], batch[positives,:]) -  self.pos_margin))
        neg_dists = torch.mean(F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors,:], batch[negatives,:])))

        loss_rank      = pos_dists + neg_dists

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
