import numpy as np
import torch
from torch import nn
from torchvision.transforms import *
from PIL import Image
import random

# np.random.seed(5)


class Reprogram(nn.Module):
    def __init__(self, model, input_size, pic_path, blur_path, pre, initial_weight=None):
        super().__init__()

        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        img = Image.open(pic_path).convert('RGB')
        trans = Compose([Resize((input_size[1], input_size[2])), ToTensor()])
        img_o = trans(img)
        self.bkg = nn.Parameter(2*img_o-1)
        self.bkg.requires_grad = False
        # added weight
        if pre == True:
            self.weight = nn.Parameter(torch.from_numpy(initial_weight))
        else:
            self.weight = nn.Parameter(torch.from_numpy(np.random.randn(input_size[0], input_size[1], input_size[2])).float())
        x = np.full(shape=(1000, 2), dtype='float32', fill_value=0)
        a = [i for i in range(0, 1000)]
        random.shuffle(a)
        for i in range(0, 1000):
            if a[i] < 500:
                x[i][0] = 1
            else:
                x[i][1] = 1
        # print(x[:,-1])
        self.out_sum = nn.Parameter(torch.from_numpy(x))
        self.out_sum.requires_grad = False
        self.slt_pix = torch.from_numpy(np.load(blur_path)).cuda()

    def forward(self, input_d):
        model_input = torch.tanh(input_d + self.bkg +
                                 torch.mul(self.slt_pix, self.weight))
        out1 = self.model(model_input)
        out = torch.matmul(out1, self.out_sum)
        # print(out1[0].shape)
        return out
