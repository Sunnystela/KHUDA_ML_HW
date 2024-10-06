import torch
import torch.nn as nn
import torch.nn.functional as F

'''

Class ArcFace
    1. def __init__(self, in_dim, out_dim, s, m):
        - s and m are parameters derived from "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
        - Matrix W:
            1) The matrix W has dimensions in_dim x out_dim.
            2) W is initialized using Xavier initialization.
            3) in_dim: Dimensionality of the tensor resulting from flattening the forward pass of VGG19.
            4) out_dim: Number of classes.
            
    2. def forward(self, x):
        - the forward pass of the ArcFace model.

'''
# ArcFace는 얼굴 인식과 같은 분류 문제에서 각도 기반 거리를 추가해서 정확도를 높이는 기술이다. 


class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        #s: 스케일링벡터, 최종 출력 확률을 조정한다
        #m: 마진 값, 분류 경계를 더 뚜렷하게 만들어 모델의 분별력을 높인다




        nn.init.kaiming_uniform_(self.W)
        
    def forward(self, x):
        normalized_x = F.normalize(x, p=2, dim=1)
        normalized_W = F.normalize(self.W, p=2, dim=0)
    
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        
        # Using torch.clamp() to ensure cosine values are within a safe range,
        # preventing potential NaN losses.
        
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        probability = self.s * torch.cos(theta+self.m)
        
        return probability
    
    #forward함수는 모델이 주어진 입력을 처리하는 순전파 경로이다
    #F.normalize(): 입력 텐서와 가중치 텐서를 각각 L2 정규화해 길이를 1로 맞춘다
    #torch.matmul(): 정규화된 입력 벡터와 가중치 행렬의 내적으로 코사인 유사도를 계산한다
    #torch.acos(): 코사인 값을 각도로 변환한다. 두 벡터 사이의 각도를 의미한다
    #torch.clamp(): 코사인 값이 범위 [-1, 1] 안에 있도록 한다.
    #self.s * torch.cos(theta + self.m): 각도에 마진을 더한 후 코사인 값을 계산하고, 이를 스케일링 팩터로 조정하여 최종 분류 확률을 구한다.

