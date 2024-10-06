from Backbone import VGG19
from ArcFace import ArcFace
import torch.nn as nn

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.ArcFace = ArcFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        x = self.ArcFace(x)
        
        return x
    
    #입력 이미지를 VGG19를 이용해 특징을 추출한뒤 arcFace를 통해 최종 예측한다.