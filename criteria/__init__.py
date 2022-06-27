### Standard DML criteria
from criteria import triplet, margin, proxynca, npair
from criteria import lifted, contrastive, softmax
from criteria import angular, snr, histogram, arcface
from criteria import softtriplet, multisimilarity, quadruplet, multisimilarity_lsd
### Non-Standard Criteria
from criteria import adversarial_separation
from criteria import triplet_lsd, triplet_lsd_cos, triplet_lsd_joint, triplet_lsd_reweight, \
    triplet_lsd_reweight_standard, triplet_lsd_reweight_allsin, triplet_lsd_reweight_check, triplet_lsd_reweight_v2
from criteria import contrastive_lsd, margin_lsd, multisimilarity_lsd, softmax_lsd, ap_loss_lsd, proxynca_lsd, ap_loss, triplet_lsd_hard
### Basic Libs
import copy


"""================================================================================================="""
def select(loss, opt, to_optim=None, batchminer=None):
    #####
    losses = {'triplet': triplet,
              'margin':margin,
              'margin_lsd': margin_lsd,
              'proxynca':proxynca,
              'proxynca_lsd':proxynca_lsd,
              'npair':npair,
              'angular':angular,
              'contrastive':contrastive,
              'contrastive_lsd':contrastive_lsd,
              'lifted':lifted,
              'snr':snr,
              'multisimilarity':multisimilarity,
              'multisimilarity_lsd': multisimilarity_lsd,
              'histogram':histogram,
              'softmax':softmax,
              'softmax_lsd':softmax_lsd,
              'softtriplet':softtriplet,
              'arcface':arcface,
              'quadruplet':quadruplet,
              'adversarial_separation':adversarial_separation,
              'triplet_lsd': triplet_lsd,
              'triplet_lsd_cos': triplet_lsd_cos,
              'triplet_lsd_joint': triplet_lsd_joint,
              'triplet_lsd_reweight': triplet_lsd_reweight,
              'triplet_lsd_reweight_standard': triplet_lsd_reweight_standard,
              'triplet_lsd_reweight_allsin': triplet_lsd_reweight_allsin,
              'triplet_lsd_reweight_check': triplet_lsd_reweight_check,
              'triplet_lsd_reweight_v2': triplet_lsd_reweight_v2,
              'triplet_lsd_hard': triplet_lsd_hard,
              'ap_loss_lsd': ap_loss_lsd,
              'ap_loss': ap_loss
              }


    if loss not in losses: raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = losses[loss]
    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(loss, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(batchminer.name, loss))


    loss_par_dict  = {'opt':opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer

    criterion = loss_lib.Criterion(**loss_par_dict)

    if to_optim is not None:
        if loss_lib.REQUIRES_OPTIM:
            if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
                to_optim += criterion.optim_dict_list
            else:
                to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]

        return criterion, to_optim
    else:
        return criterion
