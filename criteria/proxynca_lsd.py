import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()

        ####
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        self.proxies            = torch.nn.Parameter(torch.randn(self.num_proxies, self.embed_dim)/8)
        self.class_idxs         = torch.arange(self.num_proxies)

        self.name           = 'proxynca'

        self.optim_dict_list = [{'params':self.proxies, 'lr':opt.lr * opt.loss_proxynca_lrmulti}]


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
        #Empirically, multiplying the embeddings during the computation of the loss seem to allow for more stable training;
        #Acts as a temperature in the NCA objective.
        batch   = 3*torch.nn.functional.normalize(batch, dim=1)
        proxies = 3*torch.nn.functional.normalize(self.proxies, dim=1)
        #Group required proxies
        pos_proxies = torch.stack([proxies[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.class_idxs[:class_label],self.class_idxs[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([proxies[neg_labels,:] for neg_labels in neg_proxies])
        #Compute Proxy-distances
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies).pow(2),dim=-1)
        #Compute final proxy-based NCA loss
        loss_rank = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))

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
