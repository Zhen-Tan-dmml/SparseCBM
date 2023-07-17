import os
import torch
import numpy as np 
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class FC(torch.nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim=0, stddev=None):
        """
        Extend standard Torch Linear layer
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = torch.nn.ReLU()
            self.fc_new = torch.nn.Linear(input_dim, expand_dim)
            self.fc = torch.nn.Linear(expand_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = torch.nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x

class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_attributes=0,n_class_attr=0):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.n_attributes = n_attributes
        self.n_class_attr = n_class_attr

    def forward_stage2(self, stage1_out):
        if self.use_relu:
            attr_outputs = [torch.nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        # stage2_inputs = torch.cat(stage2_inputs, dim=1)
        XtoC_logits = torch.stack(stage2_inputs, dim=0)
        XtoC_logits=torch.transpose(XtoC_logits, 0, 1) #torch.Size([8, 10, 3])
        predictions_concept_labels = XtoC_logits.reshape(-1,self.n_attributes*self.n_class_attr)
        # predictions_concept_labels = torch.argmax(XtoC_logits, axis=-1)
        # predictions_concept_labels = F.one_hot(predictions_concept_labels)
        # predictions_concept_labels = predictions_concept_labels.reshape(-1,self.n_attributes*self.n_class_attr)
        # predictions_concept_labels = predictions_concept_labels.to(torch.float32)
        stage2_inputs = predictions_concept_labels
        all_out = [self.sec_model(stage2_inputs)]
        all_out.extend(stage1_out)
        return all_out

    def forward(self, x):
        if self.first_model.training:
            outputs = self.first_model(x) 
            return self.forward_stage2(outputs)
        else:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs)

class ModelXtoC(torch.nn.Module):
    def __init__(self, num_classes, n_attributes=4, bottleneck=False, expand_dim=0,connect_CY=False,Lstm=False,aux_logits=False):
        """
        Args:
        num_classes: number of main task classes
        aux_logits: whether to also output auxiliary logits
        transform input: whether to invert the transformation by ImageNet (should be set to True later on)
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        three_class: whether to count not visible as a separate class for predicting attribute
        """
        super(ModelXtoC, self).__init__()
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.all_fc = torch.nn.ModuleList() #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        self.num_classes = num_classes
        self.Lstm = Lstm
        self.aux_logits = aux_logits

        dim = 768
        if self.Lstm:
            dim = 128
        
        if self.aux_logits:
            self.AuxLogits = ModelXtoCAux(num_classes = self.num_classes, n_attributes = self.n_attributes, bottleneck = self.bottleneck, expand_dim = 0,Lstm = self.Lstm)
        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(dim, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(dim, num_classes, expand_dim))
        else:
            self.all_fc.append(FC(dim, num_classes, expand_dim))

    def forward(self, x):
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        if self.aux_logits:
            out_aux = self.AuxLogits(x)
            aux_concepts_logits = [item.cpu().detach().numpy() for item in out_aux]
            np.save('aux_concepts_logits.npy',np.array(aux_concepts_logits))
        return out

class ModelXtoCAux(torch.nn.Module):
    def __init__(self, num_classes, n_attributes, bottleneck=False, expand_dim=0, connect_CY=False,Lstm=False):
        """
        Args:
        num_classes: number of main task classes
        aux_logits: whether to also output auxiliary logits
        transform input: whether to invert the transformation by ImageNet (should be set to True later on)
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        three_class: whether to count not visible as a separate class for predicting attribute
        """
        super(ModelXtoCAux, self).__init__()
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.all_fc = torch.nn.ModuleList() #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        self.num_classes = num_classes
        self.Lstm = Lstm

        dim = 768
        if self.Lstm:
            dim = 128

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(dim, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(dim, num_classes, expand_dim))
        else:
            self.all_fc.append(FC(dim, num_classes, expand_dim))

    def forward(self, x):
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        return out