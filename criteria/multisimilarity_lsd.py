import torch, torch.nn as nn



"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.thresh     = opt.loss_multisimilarity_thresh

        self.name           = 'multisimilarity'

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
        return ls(x/tau)

    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x/tau)

    def kl_d(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(input, target)

    def forward(self, batch, teacher_batch, labels, epoch, **kwargs):
        similarity = batch.mm(batch.T)

        loss_rank = []
        for i in range(len(batch)):
            pos_idxs       = labels==labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels!=labels[i]

            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            ### This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            loss_rank.append(pos_term + neg_term)

        loss_rank = torch.mean(torch.stack(loss_rank))

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
