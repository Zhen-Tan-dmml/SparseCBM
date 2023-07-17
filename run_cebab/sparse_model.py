import os
import sys
import copy
import random
import numpy as np

import torch 
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


__all__ = ['insert_sparse_mask', 'set_required_gradient', 'set_sparse_index', 'setup_seed', 'set_mask_gradient']


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class GetSubnet_Structure(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        assert len(scores.shape) == 2 # scores (out_dim, in_dim)
        out = scores.clone()
        channel_scores = torch.norm(scores, dim=-1)
        _, idx = channel_scores.sort()
        j = int((1 - k) * channel_scores.numel())
        out[idx[:j], :] = 0
        out[idx[j:], :] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# https://github.com/inspire-group/hydra/blob/master/models/layers.py
class SubnetLinear(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight

    def __init__(self, origin_layer, num_concept, bias=True):
        super().__init__()
        self.fc = origin_layer
        self.popup_scores = nn.ParameterList([Parameter(torch.Tensor(self.fc.weight.shape)) for _ in range(num_concept)])
        for sparse_idx in range(num_concept):
            nn.init.kaiming_uniform_(self.popup_scores[sparse_idx], a=math.sqrt(5))

        self.sparse_idx = 0

    def set_prune_rate(self, k):
        self.k = k

    def set_index(self, sparse_idx):
        self.sparse_idx = sparse_idx

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores[self.sparse_idx].abs(), self.k) # self.k \in [0,1], activated ratio
        x = F.linear(x, self.fc.weight * adj, self.fc.bias)
        return x


# https://github.com/inspire-group/hydra/blob/master/models/layers.py
class SubnetLinear_Structure(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight

    def __init__(self, origin_layer, num_concept, bias=True):
        super().__init__()
        self.fc = origin_layer
        self.popup_scores = nn.ParameterList([Parameter(torch.Tensor(self.fc.weight.shape)) for _ in range(num_concept)])
        for sparse_idx in range(num_concept):
            nn.init.kaiming_uniform_(self.popup_scores[sparse_idx], a=math.sqrt(5))

        self.sparse_idx = 0

    def set_prune_rate(self, k):
        self.k = k

    def set_index(self, sparse_idx):
        self.sparse_idx = sparse_idx

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet_Structure.apply(self.popup_scores[self.sparse_idx].abs(), self.k) # self.k \in [0,1], activated ratio
        x = F.linear(x, self.fc.weight * adj, self.fc.bias)
        return x



# https://github.com/inspire-group/hydra/blob/master/models/layers.py
class SubnetLinear_Mask(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight

    def __init__(self, origin_layer, num_concept, bias=True):
        super().__init__()
        self.fc = origin_layer
        self.mask_list = nn.ParameterList([Parameter(torch.ones(self.fc.weight.shape)) for _ in range(num_concept)])
        self.sparse_idx = 0
        self.restore_mask = None

    def set_index(self, sparse_idx):
        self.sparse_idx = sparse_idx

    def modify_mask(self, ratio):

        change_num_para = int(ratio * self.fc.weight.nelement())
        current_prune_para = int(self.mask_list[self.sparse_idx].eq(0).float().sum().item())
        # magnitude pruning
        new_mask = self.mask_list[self.sparse_idx].clone()

        score_prune = self.mask_list[self.sparse_idx] * self.mask_list[self.sparse_idx].grad * self.fc.weight
        score_prune = score_prune.abs().detach().clone()
        remove_para = current_prune_para + change_num_para
        _, idx = score_prune.flatten().sort()
        flat_new_mask = new_mask.flatten()
        flat_new_mask[idx[:remove_para]] = 0
        flat_new_mask[idx[remove_para:]] = 1

        # gradient grow
        score_grow = (1 - new_mask) * self.mask_list[self.sparse_idx].grad * self.fc.weight + 1e+10 * new_mask
        score_grow = score_grow.abs().detach().clone()
        _, idx = score_grow.flatten().sort()
        flat_new_mask = new_mask.flatten()
        flat_new_mask[idx[:current_prune_para]] = 0
        flat_new_mask[idx[current_prune_para:]] = 1

        # check mask
        self.restore_mask = self.mask_list[self.sparse_idx].data.clone()
        self.mask_list[self.sparse_idx].data = new_mask

    def recover_mask(self):
        self.mask_list[self.sparse_idx].data = self.restore_mask
        self.restore_mask = None

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        x = F.linear(x, self.fc.weight * self.mask_list[self.sparse_idx], self.fc.bias)
        return x






def convert_to_multi_concept_sparse_model(model, num_concept, sparse_ratio): # Consider all linear layers

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_to_multi_concept_sparse_model(module, num_concept, sparse_ratio)

        if isinstance(module, nn.Linear):

            if model._modules[name].bias is None:
                bias = False
            else:
                bias = True
            model._modules[name] = SubnetLinear(model._modules[name], num_concept, bias)
            model._modules[name].set_prune_rate(sparse_ratio)
    return model

def convert_to_multi_concept_sparse_model_structure(model, num_concept, sparse_ratio): # Consider all linear layers

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_to_multi_concept_sparse_model_structure(module, num_concept, sparse_ratio)

        if isinstance(module, nn.Linear):

            if model._modules[name].bias is None:
                bias = False
            else:
                bias = True
            model._modules[name] = SubnetLinear_Structure(model._modules[name], num_concept, bias)
            model._modules[name].set_prune_rate(sparse_ratio)
    return model

def convert_to_multi_concept_sparse_model_with_custom_mask(model, num_concept): # Consider all linear layers

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_to_multi_concept_sparse_model_with_custom_mask(module, num_concept)

        if isinstance(module, nn.Linear):

            if model._modules[name].bias is None:
                bias = False
            else:
                bias = True
            model._modules[name] = SubnetLinear_Mask(model._modules[name], num_concept, bias)
    return model


## Convert to Sparse Model
def insert_sparse_mask(model, num_concept, sparse_ratio, pruning_type='unstructure'):

    # pretrained_checkpoint = copy.deepcopy(model.state_dict())

    if pruning_type == 'unstructure':
        model = convert_to_multi_concept_sparse_model(model, num_concept, sparse_ratio)
    elif pruning_type == 'structure':
        model = convert_to_multi_concept_sparse_model_structure(model, num_concept, sparse_ratio)
    elif pruning_type == 'obert':
        print('Pruning with OBERT')
        model = convert_to_multi_concept_sparse_model_with_custom_mask(model, num_concept)
        for name, module in model.named_modules():
            if isinstance(module, SubnetLinear_Mask):
                for p in module.mask_list:
                    p.requires_grad = False
                print('{}: mask_grad: False'.format(name))
    else:
        raise ValueError('Unsupported Pruning Type')

    # ## Load original checkpoint
    # for key in model.state_dict().keys():
    #     if not key in pretrained_checkpoint:
    #         print('Add new weight {}'.format(key))
    #         pretrained_checkpoint[key] = model.state_dict()[key]
    # model.load_state_dict(pretrained_checkpoint)

    return model


### Set gradient requirements True or False
def set_required_gradient(model, parameter_grad=True, score_grad=True):

    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear):
            module.fc.weight.requires_grad = parameter_grad
            if not module.fc.bias is None:
                module.fc.bias.requires_grad = parameter_grad
            for p in module.popup_scores:
                p.requires_grad = score_grad
            print('{}: parameter_grad {}, score_grad {}'.format(name, parameter_grad, score_grad))
    return model

def set_mask_gradient(model):
    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear_Mask):
            module.fc.weight.requires_grad = False
            if not module.fc.bias is None:
                module.fc.bias.requires_grad = False
            for p in module.mask_list:
                p.requires_grad = True
    return model


def set_required_gradient_structure(model, parameter_grad=True, score_grad=True):

    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear_Structure):
            module.fc.weight.requires_grad = parameter_grad
            if not module.fc.bias is None:
                module.fc.bias.requires_grad = parameter_grad
            for p in module.popup_scores:
                p.requires_grad = score_grad
            print('{}: parameter_grad {}, score_grad {}'.format(name, parameter_grad, score_grad))
    return model

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


### Set required sparse index
def set_sparse_index(model, index):
    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear):
            module.set_index(index)


if __name__ == '__main__':
    from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    # insert sparse module
    model = insert_sparse_mask(model, 5, 0.1)
    # set required_gradient
    model = set_required_gradient(model, True, False)
    model = set_required_gradient(model, False, True)

    # # Forward Test
    # x = 'As I sit here on my porch, sipping my coffee and watching the world go by, I can not help but feel a sense of wonder at the sheer complexity of everything around us. From the smallest particle to the grandest galaxy, the universe is a tapestry of infinite detail and beauty. And yet, for all its complexity, there is a simplicity to it all that is truly awe-inspiring. Everything is connected, in ways that we can not even begin to fathom. Every action has a reaction, every cause has an effect. And yet, even with all the knowledge that we have amassed, there is still so much that we do not understand. There are mysteries that have eluded us for centuries, and may continue to do so for centuries to come. But that does not stop us from trying to unravel them. It does not stop us from exploring the depths of our own consciousness, or the vast expanse of the cosmos. It does not stop us from seeking answers to the biggest questions of all. Who are we? Why are we here? What is the meaning of life? These are questions that have plagued us since the dawn of time, and yet we continue to search for answers. Perhaps it is in the search itself that we find meaning. Perhaps it is in the journey, rather than the destination, that we discover the true nature of our existence. And so, as I sit here on my porch, watching the world go by, I am content to simply marvel at the beauty and complexity of it all, and to embrace the mystery that lies at the heart of our being.'
    # input_data = tokenizer(x)

    # import pdb; pdb.set_trace()
    # set_sparse_index(model, 0)    
    # outputs = model(input_ids=torch.tensor(input_data['input_ids']).reshape(1,-1), attention_mask=torch.tensor(input_data['attention_mask']).reshape(1,-1))
    # print(outputs['pooler_output'][0, :10])

    # import pdb; pdb.set_trace()
    # set_sparse_index(model, 1)    
    # outputs = model(input_ids=torch.tensor(input_data['input_ids']).reshape(1,-1), attention_mask=torch.tensor(input_data['attention_mask']).reshape(1,-1))
    # print(outputs['pooler_output'][0, :10])

    # import pdb; pdb.set_trace()
    # set_sparse_index(model, 2)    
    # outputs = model(input_ids=torch.tensor(input_data['input_ids']).reshape(1,-1), attention_mask=torch.tensor(input_data['attention_mask']).reshape(1,-1))
    # print(outputs['pooler_output'][0, :10])

    # import pdb; pdb.set_trace()
    # set_sparse_index(model, 3)    
    # outputs = model(input_ids=torch.tensor(input_data['input_ids']).reshape(1,-1), attention_mask=torch.tensor(input_data['attention_mask']).reshape(1,-1))
    # print(outputs['pooler_output'][0, :10])

