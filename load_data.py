import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


np.random.seed(5)


def load_data(proportion, data_path, posi_path, resize=(3, 224, 224), shuffle=True, theta=1.):
    data_np = np.load(data_path)
    if shuffle:
        np.random.shuffle(data_np)
    samples = np.split(data_np, [379], axis=1)[0].astype('float32')
    labels_out = np.squeeze(np.split(data_np, [379, 380], axis=1)[1])
    if proportion != 1:
        samples = np.split(samples, [int(len(samples) * proportion)], axis=0)[0]
        labels_out = np.split(labels_out, [int(len(labels_out) * proportion)], axis=0)[0]

    samples_out = np.zeros((int(len(samples)), resize[0], resize[1], resize[2]), dtype='float32')
    scheme = np.load(posi_path)
    # print(samples.shape)
    # print(scheme.shape)
    if scheme.shape[0] != samples.shape[1]:
        raise Exception("placement is illegal!")
    # print("theta:", theta)
    for k in range(0, samples.shape[0]):
        for i in range(0, samples.shape[1]):
            if samples[k][i] == 1:
                samples_out[k][0][scheme[i][0]][scheme[i][1]] = -theta
                samples_out[k][1][scheme[i][0]][scheme[i][1]] = -theta
                samples_out[k][2][scheme[i][0]][scheme[i][1]] = -theta
            if samples[k][i] == 0:
                samples_out[k][0][scheme[i][0]][scheme[i][1]] = theta
                samples_out[k][1][scheme[i][0]][scheme[i][1]] = theta
                samples_out[k][2][scheme[i][0]][scheme[i][1]] = theta

    return samples_out, labels_out


def loader_create(proportion, data_path, posi_path, batch_size=16, resize = (3, 224, 224), shuffle=True, theta=1.):
    # all samples are 27153, malignant is 7323, benign is 19830, dimension is 379
    # benign label is 0 and malignant label is 1
    data, label = load_data(proportion=proportion, data_path=data_path, posi_path=posi_path,
                            resize=resize, shuffle=shuffle, theta=theta)

    [train_data, test_data] = np.split(data, [int(0.7 * len(data))], axis=0)
    [train_label, test_label] = np.split(label, [int(0.7 * len(label))], axis=0)
    # print(train_data.shape)
    # print(test_data.shape)

    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_label).long())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader