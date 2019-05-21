import numpy as np
import scipy.stats as scipystats
import torch.nn as nn
import torch

import os
import matplotlib.pyplot as plt
import scipy.misc as sm
import cv2
import random

def get_1x(model):
    b = []
    name = []
    b.append(model.module.Scale.conv1)
    b.append(model.module.Scale.bn1)
    b.append(model.module.Scale.layer1)
    b.append(model.module.Scale.layer2)
    b.append(model.module.Scale.layer3)
    b.append(model.module.Scale.layer4)

    name.append('Scale.conv1')
    name.append('Scale.bn1')
    name.append('Scale.layer1')
    name.append('Scale.layer2')
    name.append('Scale.layer3')
    name.append('Scale.layer4')

    for i in range(len(b)):
        for indj, j in enumerate(b[i].modules()):
#            print('---')
#            print(j)
            jj = 0
            for indk, k in enumerate(j.parameters()):
#                print('----------')
#                print(k)
                jj += 1
                if k.requires_grad:
                    #yield k
                    yield [name[i]+'-'+str(indj)+'-'+str(indk),k]


def get_10x(model):

    b = []
    b.append(model.module.Scale.layer5.parameters())

    for j in range(len(b)):
        for index,i in enumerate(b[j]):
            yield [str(j)+'-'+str(index)+'-layer5',i]
