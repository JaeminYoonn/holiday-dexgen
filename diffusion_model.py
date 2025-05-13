import torch
import torch.nn as nn
import math


class FiLM(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return (1 + gamma) * x + beta


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=6, group_size=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.ReLU()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append(nn.GroupNorm(group_size, hidden_dim))

        self.output_dim = hidden_dim

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = self.act(norm(layer(x)))
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)


class UNetEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, cond_dim=1024):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.film = FiLM(hidden_dim, cond_dim)
        self.act = nn.ReLU()

    def forward(self, x, cond):
        x = self.act(self.fc(x))
        x = self.film(x, cond)
        return


class UNetDecoderBlock(nn.Module):
    def __init__(self, output_dim, hidden_dim=768, cond_dim=1024):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.film = FiLM(output_dim, cond_dim)
        self.act = nn.ReLU()

    def forward(self, x, cond):
        x = self.act(self.fc(x))
        x = self.film(x, cond)
        return x


class UNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, depth=3, cond_dim=1024):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                UNetEncoderBlock(
                    input_dim if i == 0 else hidden_dim, hidden_dim, cond_dim
                )
                for i in range(depth)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                UNetDecoderBlock(
                    hidden_dim if i < depth - 1 else input_dim, hidden_dim, cond_dim
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, cond):
        # encoding phase
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x, cond)
            if i < len(self.encoders) - 1:
                skips.append(x)

        # decoding phase
        for i, dec in enumerate(self.decoders):
            if i > 0:
                x = x + skips.pop()
            x = dec(x, cond)

        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        input_dim,
        state_dim,
        time_hidden_dim=1024,
        mlp_hidden_dim=1024,
        mlp_num_layers=6,
        mlp_group_size=8,
        unet_hidden_dim=768,
        unet_depth=3,
        cond_dim=1024,
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(time_hidden_dim)
        self.state_encoder = MLPEncoder(
            state_dim, mlp_hidden_dim, mlp_num_layers, mlp_group_size
        )

        proj_input_dim = self.time_encoder.output_dim + self.state_encoder.output_dim
        self.project = nn.Linear(proj_input_dim, cond_dim)

        self.unet = UNet(
            input_dim=input_dim,
            hidden_dim=unet_hidden_dim,
            depth=unet_depth,
            cond_dim=cond_dim,
        )

    def forward(self, noised_motion, timestep, state):
        time_feat = self.time_encoder(timestep)
        state_feat = self.state_encoder(state)

        cond = self.project(torch.cat([time_feat, state_feat], dim=-1))
        return self.unet(noised_motion, cond)
