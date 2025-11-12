import torch
import torch.nn as nn
from torchsummary import summary
from collections import namedtuple
import numpy as np
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from math import dist


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)#[1]  # (batch_size, num_points, k)
    return idx

def rand_radius_or_knn(x, k, d):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    print(pairwise_distance)
    randVals = torch.rand(pairwise_distance.shape)
    modified_distance = torch.where(pairwise_distance > -d, -randVals, pairwise_distance)
    test = modified_distance[0]
    zeros = torch.zeros((pairwise_distance.shape[-1]))
    print(modified_distance)
    test[range(pairwise_distance.shape[-1]), range(pairwise_distance.shape[-1])] = zeros
    modified_distance[0] = test
    print(modified_distance)
    idx = modified_distance.topk(k=k, dim=-1)#[1]  # (batch_size, num_points, k)
    return idx

d = 100
with open('D:/DevPython/PlantSegNet/datasets/npcs/train/0_smlm_dataset.ply', 'rb') as f:
    plydata = PlyData.read(f)
points_np = np.asarray(np.array(plydata.elements[0].data).tolist())
idx = np.random.randint(points_np.shape[0], size=25)
print(idx)
points = np.array([points_np[idx,:]])
print(points)
points = np.array([points_np])
pointst =  torch.from_numpy(points).float()
pointst = pointst.transpose(1, 2)
res = rand_radius_or_knn(pointst, 100, d*d)
print(res)
idx = res[1][0]
with open('D:/DataPALMSTORM/Tracking/somefile.txt', 'a') as the_file:
    the_file.write('Width\tHeight\tnb_Planes\tnb_Tracks\tPixel_Size(um)\tFrame_Duration(s)\tGaussian_Fit\tSpectral\n')
    the_file.write('0\t0\t1\t' + str(idx.shape[0]) + '\t1\t0\t0\t0\n')
    the_file.write('Track\tPlane\tCentroidX(px)\tCentroidY(px)\tCentroidZ(um)\tIntegrated_Intensity\tid\tPair_Distance(px)\n')
    cur_track = 1
    for x in range(idx.shape[0]):
        for y in range(idx.shape[1]):
            id = idx[x][y].numpy()
            the_file.write(str(cur_track) + '\t1\t' + str(points_np[id][0]) + '\t' + str(points_np[id][1]) + '\t0\t0\t' + str(id) + '\t0\n')
        cur_track = cur_track + 1
    