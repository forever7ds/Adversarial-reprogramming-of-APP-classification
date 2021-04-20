import time
import torch.nn
import torch
import torch.optim as optim
import numpy as np
from torchvision.models import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from model import Reprogram
from load_data import *

Mapping = torch.from_numpy(np.array([0, 1])).long() #143, 689

def train_and_test(config, proportion, posi_path, blur_path, pic_path, data_path, red_path, pre_path=None):
    if config.Classifier == "resnet18":
        model = resnet18(pretrained=True)
    elif config.Classifier == "resnet50":
        model = resnet50(pretrained=True)
    elif config.Classifier == "densenet121":
        model = densenet121(pretrained=True)
    elif config.Classifier == "inception_v3":
        model = inception_v3(pretrained=True)
    if pre_path != None:
        pre_weight = np.load(pre_path)
        net = Reprogram(model=model, input_size=config.ImageSize, pic_path=pic_path, blur_path=blur_path, pre=True, initial_weight=pre_weight).cuda()
    else:
        net = Reprogram(model=model, input_size=config.ImageSize, pic_path=pic_path, blur_path=blur_path, pre=False).cuda()
    train_loader, test_loader = loader_create(proportion=proportion, data_path=data_path, posi_path=posi_path,
                                              batch_size=config.BatchSize, theta=config.Theta, resize=config.ImageSize)
    optimizer = optim.Adam([net.weight], lr=0.01, weight_decay=1e-4)
    lr_decay = 0.9

    print("start training !")
    start_time = time.time()
    
    for epoch in range(config.Epochs):
        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        net.train()
        if epoch % 2 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= lr_decay

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            outputs = net(inputs)
            # print(outputs[0].shape)

            # Update network parameters via backpropagation: forward + backward + optimize
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, Mapping[labels].cuda())
            # loss = criterion(outputs[0], Mapping[labels].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        if (epoch >= 95 or epoch % 10 == 0) and epoch > 0:
            test_start_time = time.time()
            print("start testing ! ")
            result_y, label_y = acc_cal(test_loader, net)
            target_name = ['false', 'true']
            ts = classification_report(label_y, result_y, target_names=target_name, digits=4)
            file_handle=open(red_path+str(epoch+1)+'.txt', mode="w")
            file_handle.write(ts)
            file_handle.close()
            print(ts)
            print("test time :", time.time()-test_start_time)
            print("testing finished !")
        epoch_train_time = time.time() - epoch_start_time
        print("Epoch :", epoch + 1, "time :", epoch_train_time, "loss :", loss_epoch / n_batches)

    train_time = time.time() - start_time
    print("training and test time :", train_time)

    return net


def acc_cal(loader, model):
    model.eval()
    results = []
    labels = []
    num = 0
    for i, data in enumerate(loader):
        input_data, label_data = data
        # print(input_data.shape)
        out = model(input_data.cuda()).cpu()
        # out = torch.matmul(model(input_data.cuda()),model.out_sum).cpu()
        result = torch.argmax(out, dim=1).numpy()
        label = label_data.numpy()
        for j, k in zip(result, label):
            results.append(j)
            # for hard coded mapping
            '''if j == Mapping[0]:
                results.append(0)
            elif j == Mapping[1]:
                results.append(1)
            else:
                results.append(random.randint(0, 1))
                num += 1'''
            labels.append(k)
    print(num)
    return results, labels