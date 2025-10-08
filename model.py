import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepCABAC

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs):
        super().__init__()
        freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.embed_funcs = []
        self.embed_funcs.append(lambda x: x)
        for freq in freq_bands:
            self.embed_funcs.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_funcs.append(lambda x, freq=freq: torch.cos(x * freq))
        self.input_dim = input_dim
        self.output_dim = input_dim * len(self.embed_funcs)

    def forward(self, inputs):
        outputs = torch.cat([func(inputs) for func in self.embed_funcs], dim=-1)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        block_dim = kwargs['block_dim']
        hidden_dim = kwargs['hidden_dim']
        layer_norm = kwargs['layer_norm']
        short_cut = kwargs['short_cut']
        sine_freq = kwargs['sine_freq']
        self.linear1 = nn.Linear(block_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, block_dim)
        if layer_norm:
            self.norm = nn.LayerNorm((block_dim,))
        else:
            self.register_buffer('norm', None)
        self.short_cut = short_cut
        self.sine_freq = sine_freq

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        if self.sine_freq == 0:
            outputs = F.relu(outputs)
        else:
            outputs = torch.sin(self.sine_freq * outputs)
        outputs = self.linear2(outputs)
        if self.norm:
            outputs = self.norm(outputs)
        if self.short_cut:
            outputs = outputs + inputs
        outputs = F.relu(outputs)
        return outputs

class ResidualNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_dim = kwargs['input_dim']
        block_dim = kwargs['block_dim']
        output_dim = kwargs['output_dim']
        num_blocks = kwargs['num_blocks']
        layer_norm = kwargs['layer_norm']
        self.linear1 = nn.Linear(input_dim, block_dim)
        self.linear2 = nn.Linear(block_dim, output_dim)
        self.blocks = nn.ModuleList([ResidualBlock(**kwargs) for _ in range(num_blocks)])
        if layer_norm:
            self.norm = nn.LayerNorm((block_dim,))
        else:
            self.register_buffer('norm', None)
  
    def forward(self, inputs):
        outputs = self.linear1(inputs)
        if self.norm:
            outputs = self.norm(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.linear2(outputs)
        return outputs

class Representation(nn.Module):
    def __init__(self, step_size):
        super().__init__()
        self.step_size = step_size
    
    def forward(self, inputs):
        pass

    def l1_penalty(self, reference_model=None):
        penalty = 0
        if reference_model:
            for param, reference_param in zip(self.parameters(), reference_model.parameters()):
                penalty += torch.abs(param - reference_param).sum()
        else:
            for param in self.parameters():
                penalty += torch.abs(param).sum()
        return penalty

    @torch.no_grad()
    def sparsity(self, reference_model=None):
        num_weights = 0
        num_zeros = 0
        if reference_model:
            for param, reference_param in zip(self.parameters(), reference_model.parameters()):
                num_weights += param.numel()
                num_zeros += (torch.abs(param - reference_param) < self.step_size / 2).sum().item()
        else:
            for param in self.parameters():
                num_weights += param.numel()
                num_zeros += (torch.abs(param) < self.step_size / 2).sum().item()
        return num_zeros / num_weights
    
    def encode(self, reference_model=None):
        encoder = deepCABAC.Encoder()
        if reference_model:
            for param, reference_param in zip(self.parameters(), reference_model.parameters()):
                param = param.data.cpu().numpy()
                reference_param = reference_param.data.cpu().numpy()
                encoder.encodeWeightsRD(param - reference_param, 0.0, self.step_size, 0.0)
        else:
            for param in self.parameters():
                param = param.data.cpu().numpy()
                encoder.encodeWeightsRD(param, 0.0, self.step_size, 0.0)
        stream = encoder.finish().tobytes()
        return stream

    def decode(self, stream, reference_model=None):
        decoder = deepCABAC.Decoder()
        decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
        if reference_model:
            for param, reference_param in zip(self.parameters(), reference_model.parameters()):
                decoded_param = torch.tensor(decoder.decodeWeights(), device=param.device)
                param.data.copy_(decoded_param + reference_param)
        else:
            for param in self.parameters():
                decoded_param = torch.tensor(decoder.decodeWeights(), device=param.device)
                param.data.copy_(decoded_param)
        decoder.finish()

class Representation3D(Representation):
    def __init__(self, **kwargs):
        step_size = kwargs['step_size']
        num_freqs = kwargs['num_freqs']
        super().__init__(step_size)
        input_dim = (2 * num_freqs + 1) * 3
        if num_freqs > 0:
            self.encoding = PositionalEncoding(3, num_freqs)
        else:
            self.encoding = nn.Identity()
        self.net = ResidualNet(input_dim=input_dim, **kwargs)

    def forward(self, inputs):
        outputs = self.encoding(inputs)
        outputs = self.net(outputs)
        outputs = torch.sigmoid(outputs)
        return outputs

class Representation4D(Representation):
    def __init__(self, **kwargs):
        step_size = kwargs['step_size']
        num_freqs = kwargs['num_freqs']
        temp_freqs = kwargs['temp_freqs']
        super().__init__(step_size)
        if num_freqs > 0:
            self.encoding = PositionalEncoding(3, num_freqs)
        else:
            self.encoding = nn.Identity()
        if temp_freqs > 0:
            self.temp_encoding = PositionalEncoding(1, temp_freqs)
        else:
            self.temp_encoding = nn.Identity()
        input_dim = (2 * num_freqs + 1) * 3 + (2 * temp_freqs + 1) * 1
        self.net = ResidualNet(input_dim=input_dim, **kwargs)

    def forward(self, inputs):
        outputs = self.encoding(inputs[:, 1:])
        temp_outputs = self.temp_encoding(inputs[:, :1])
        outputs = torch.cat([temp_outputs, outputs], axis=-1)
        outputs = self.net(outputs)
        outputs = torch.sigmoid(outputs)
        return outputs

def get_model(component, configs, device):
    assert component in ['geometry', 'attribute']
    if component == 'geometry':
        output_dim = 1
    elif component == 'attribute':
        output_dim = 3
    kwargs = {
        'num_freqs': configs[component]['num_freqs'],
        'temp_freqs': configs[component]['temp_freqs'],
        'block_dim': configs[component]['block_dim'],
        'hidden_dim': configs[component]['hidden_dim'],
        'output_dim': output_dim,
        'num_blocks': configs[component]['num_blocks'],
        'layer_norm': configs[component]['layer_norm'],
        'short_cut': configs[component]['short_cut'],
        'step_size': 1 / configs[component]['quantization_steps'],
        'sine_freq': configs[component]['sine_freq']
    }
    group_size = configs['group_size']
    if group_size == 1:
        model = Representation3D(**kwargs).to(device)
    else:
        model = Representation4D(**kwargs).to(device)
    return model