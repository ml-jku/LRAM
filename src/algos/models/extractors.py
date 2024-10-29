import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp


class FlattenExtractorWithMLP(FlattenExtractor):

    def __init__(self, observation_space, net_arch=None):
        super().__init__(observation_space)
        if net_arch is None:
            net_arch = [128, 128]

        mlp = create_mlp(self.features_dim, net_arch[-1], net_arch)
        self.mlp = nn.Sequential(*mlp)
        self._features_dim = net_arch[-1]

    def forward(self, observations):
        return self.mlp(self.flatten(observations))
    

class TextureFeatureExtractor(BaseFeaturesExtractor):
    """
    Textures Feature Extractor for Crafter. Textures that at dim 21
    """
    def __init__(self, observation_space, features_dim=256, texture_start_dim=21, num_textures=63,
                 texture_embed_dim=4, textures_shape=(9,7), hidden_dim=192, **kwargs):
        super().__init__(observation_space, features_dim=features_dim)
        self.texture_start_dim = texture_start_dim
        self.texture_emb = nn.Embedding(num_textures + 1, texture_embed_dim)
        self.texture_net = nn.Sequential(
            nn.Linear(texture_embed_dim * textures_shape[0] * textures_shape[1], hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.out = nn.Linear(texture_start_dim + hidden_dim, self.features_dim)
        
    def forward(self, observations):
        # receives flatttened info + textures
        info, textures = observations[..., :self.texture_start_dim], observations[..., self.texture_start_dim:].long()
        texture_embeds = self.texture_emb(textures).flatten(-2)
        texture_features = self.texture_net(texture_embeds)
        x = torch.cat((info, texture_features), dim=-1)   
        return self.out(x)


def create_cwnet(
    input_dim: int,
    output_dim: int,
    net_arch=(256,256,256),
    # activation_fn=lambda: nn.LeakyReLU(negative_slope=0.2),
    activation_fn=nn.LeakyReLU,
    squash_output: bool = False,
):
    """
    Creates the same Net as described in https://arxiv.org/pdf/2105.10919.pdf
    Basically just adds LayerNorm + Tanh after first Dense layer.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0])]
    else:
        modules = []

    modules.append(nn.LayerNorm(net_arch[0]))
    modules.append(nn.Tanh())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
