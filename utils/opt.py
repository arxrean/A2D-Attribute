import torch.nn as nn
import torch
import torch.optim as optim


def get_optimizer(args, model, epoch):
    lr = args.lr*(args.lr_dec**epoch)

    return optim.Adam([{'params': model.parameters(), 'lr': lr}], weight_decay=args.wt_dec)


def get_finetune_optimizer(args, model):
    lr = args.lr
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list = []
    for name, value in model.named_parameters():
        if 'fc' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    assert sum([len(weight_list), len(bias_list), len(last_weight_list),
                len(last_bias_list)]) == len(list(model.named_parameters()))

    return optim.Adam([{'params': weight_list, 'lr': lr},
                       {'params': bias_list, 'lr': lr*2},
                       {'params': last_weight_list, 'lr': lr*10},
                       {'params': last_bias_list, 'lr': lr*20}], weight_decay=args.wt_dec)


def get_op_optimizer(args, model):
    attr_params = [param for name, param in model.named_parameters(
    ) if 'attr_op' in name and param.requires_grad]
    other_params = [param for name, param in model.named_parameters(
    ) if 'attr_op' not in name and param.requires_grad]
    optim_params = [{'params': attr_params,
                     'lr': 0.1*args.lr}, {'params': other_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wt_dec)

    return optimizer
