import base64
import io
import torch
from PIL import Image
from torchvision import transforms
import math
import json
import sys

import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

#from config import device, num_classes, emb_size

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from torch import linalg


#model_path="./CustomArc_efficientnet.pth"
model_weight_path="CustomArc_efficientnet.pth"

input_data = "${inputData}"

image = Image.open(io.BytesIO(base64.b64decode(input_data)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.s = scale
        self.sin_m = torch.sin(torch.tensor(margin))
        self.cos_m = torch.cos(torch.tensor(margin))
        self.cout = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        cos = self.fc(x) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            one_hot = F.one_hot(label, num_classes=self.cout)
            sin = (1 - cos ** 2) ** 0.5
            angle_sum = cos * cos_m - sin * sin_m
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            cos = cos * self.s

        return cos

class CustomArc_efficientnet(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super(CustomArc_efficientnet, self).__init__()
        self.device = device
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.arc_margin_product = ArcMarginProduct(1280, num_classes, device=self.device,scale=32.0)
        nn.init.kaiming_normal_(self.arc_margin_product.weight)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x, labels=None):

        features = self.backbone.extract_features(x)

        features=self.avgpool(features)

        features = F.normalize(features)

        features = features.view(features.size(0), -1)

        features = self.arc_margin_product(features, labels)

        return features
    


class CustomArc_efficientnet(torch.nn.Module):
    def __init__(self):
        super(CustomArc_efficientnet, self).__init__()
        # 여기에 모델의 구조를 정의해야 합니다.
        # 예시로 간단하게 선언했습니다.

    def forward(self, x):
        # 모델의 forward 연산을 정의해야 합니다.
        # 예시로 간단하게 선언했습니다.
        return x

# 입력에 대한 예측 수행 함수
def predict(model, input_data):
    input_tensor = torch.tensor([[input_data]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

# 모델 인스턴스 생성 및 학습된 가중치 불러오기
def load_model(model,model_weight_path):
    model = torch.load(model)
    model.load_state_dict(torch.load(model_weight_path),map_location=torch.device('cpu'))
    model.eval()
    return model

if __name__ == "__main__":
    try:
        image_data=json.loads(sys.stdin.read())
        model=CustomArc_efficientnet(num_classes=5)
        model = load_model(model,model_weight_path)
        
        model.eval()
        transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
        input_tensor = transform(image_data).unsqueeze(0)
        # 모델에 입력하여 예측 수행
        with torch.no_grad():
            output = model(input_tensor)
        prediction = output[0].item()
        sys.stdout.write(json.dumps(prediction))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)

