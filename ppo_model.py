import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.nn import MultiheadAttention
from torch.nn import TransformerDecoderLayer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Custom_Policy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        routing_info_width = 4
        proximity_info_width = 4

        self.decoder = TransformerDecoderLayer(
            d_model=routing_info_width,
            dim_feedforward=16,
            nhead=1,
            activation="relu")

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))

        self.fc1 = nn.Linear(800, 64)
        self.fc2 = nn.Linear(64, proximity_info_width)

        self.self_attn_fusion = MultiheadAttention(
            embed_dim=8, num_heads=1)
        self.ReLU = nn.ReLU()

    def forward(self, features):
        a = features['numerical']
        b = features['image']

        a = a.unsqueeze(0) if a.dim() == 2 else a
        a = self.decoder(tgt=a, memory=a)
        a = a.squeeze(0) if a.size(0) == 1 else a

        b = b.unsqueeze(1)
        b = self.cnn(b)
        b = b.view(b.size(0), -1)
        b = F.relu(self.fc1(b))
        b = self.fc2(b)

        c = torch.cat((a, b), dim=1)
        c = self.ReLU(self.self_attn_fusion(c, c, c)[0])

        return c


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=8)
        self.extractor = Custom_Policy()

    def forward(self, observations):
        return self.extractor(observations)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy,self).__init__(*args,**kwargs,
            features_extractor_class=CustomExtractor,
            features_extractor_kwargs=dict())
