import itertools
import torch


def named_grad_param(model, keys):
    '''
        Return a generator that generates learnable named parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p[1].requires_grad,
                model[keys[0]].named_parameters())
    else:
        return filter(lambda p: p[1].requires_grad,
                itertools.chain.from_iterable(
                    model[key].named_parameters() for key in keys))


def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))


def get_norm(model):
    '''
        Compute norm of the gradients
    '''
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm()
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm


def load_model_state_dict(model, model_path):
    # 初始化模型参数
    model_dict = model.state_dict()                                    # 取出自己网络的参数字典
    pretrained_dict = torch.load(model_path)# 加载预训练网络的参数字典
    # 取出预训练网络的参数字典
    keys = []
    for k, v in pretrained_dict.items():
           keys.append(k)

    i = 0

    # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
    print("_____________pretrain_parameters______________________________")
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            print(model_dict[k])
            i = i + 1
        # print(model_dict[k])
    print("___________________________________________________________")
    model.load_state_dict(model_dict)
    return model
