import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
import deepCABAC
from model import PositionalEncoding, Representation3D

class BezierCoefficient(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.binom = torch.Tensor(binom(num_nodes - 1, np.arange(num_nodes), dtype=np.float32))
        self.range = torch.arange(num_nodes, dtype=torch.float32)
        self.reversed_range = torch.arange(num_nodes - 1, -1, -1, dtype=torch.float32)

    def forward(self, t):
        return self.binom * torch.pow(t, self.range) * torch.pow((1 - t), self.reversed_range)

class ModuleCurve(nn.Module):
    def __init__(self, fix_nodes):
        super().__init__()
        self.fix_nodes = fix_nodes
        self.num_nodes = len(fix_nodes)
        self.weights = None
        self.biases = None

    def reset_parameters(self):
        pass
    
    def compute_parameters_t(self, coeffs):
        weight_t = None
        bias_t = None
        if self.weights is not None:
            for weight, coeff in zip(self.weights, coeffs):
                if weight_t is None:
                    weight_t = weight * coeff
                else:
                    weight_t += weight * coeff
        if self.biases is not None:
            for bias, coeff in zip(self.biases, coeffs):
                if bias_t is None:
                    bias_t = bias * coeff
                else:
                    bias_t += bias * coeff
        return weight_t, bias_t

class LinearCurve(ModuleCurve):
    def __init__(self, fix_nodes, in_features, out_features, bias=True):
        super().__init__(fix_nodes)
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.ParameterList([nn.Parameter(torch.empty((out_features, in_features)), requires_grad=not fix) for fix in fix_nodes])
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.empty((out_features,)), requires_grad=not fix) for fix in fix_nodes])
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=np.sqrt(5))
        if self.biases is not None:
            for weight, bias in zip(self.weights, self.biases):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    def forward(self, inputs, coeffs):
        weight_t, bias_t = self.compute_parameters_t(coeffs)
        return F.linear(inputs, weight_t, bias_t)

class LayerNormCurve(ModuleCurve):
    def __init__(self, fix_nodes, normalized_shape, eps=1e-5, bias=True):
        super().__init__(fix_nodes)
        self.normalized_shape = normalized_shape 
        self.eps = eps
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(self.normalized_shape), requires_grad=not fix) for fix in fix_nodes])
        if bias:
            self.biases = nn.ParameterList([nn.Parameter(torch.empty(self.normalized_shape), requires_grad=not fix) for fix in fix_nodes])
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.ones_(weight)
        if self.biases is not None:
            for bias in self.biases:
                nn.init.zeros_(bias)

    def forward(self, inputs, coeffs):
        weight_t, bias_t = self.compute_parameters_t(coeffs)
        return F.layer_norm(inputs, self.normalized_shape, weight_t, bias_t, self.eps)

class ResidualBlockCurve(nn.Module):
    def __init__(self, fix_nodes, **kwargs):
        super().__init__()
        block_dim = kwargs['block_dim']
        hidden_dim = kwargs['hidden_dim']
        layer_norm = kwargs['layer_norm']
        short_cut = kwargs['short_cut']
        sine_freq = kwargs['sine_freq']
        self.linear1 = LinearCurve(fix_nodes, block_dim, hidden_dim)
        self.linear2 = LinearCurve(fix_nodes, hidden_dim, block_dim)
        if layer_norm:
            self.norm = LayerNormCurve(fix_nodes, (block_dim,))
        else:
            self.register_buffer('norm', None)
        self.short_cut = short_cut
        self.sine_freq = sine_freq

    def forward(self, inputs, coeffs):
        outputs = self.linear1(inputs, coeffs)
        if self.sine_freq == 0:
            outputs = F.relu(outputs)
        else:
            outputs = torch.sin(self.sine_freq * outputs)
        outputs = self.linear2(outputs, coeffs)
        if self.norm:
            outputs = self.norm(outputs, coeffs)
        outputs = F.relu(outputs + inputs if self.short_cut else outputs)
        return outputs

class ResidualNetCurve(nn.Module):
    def __init__(self, fix_nodes, **kwargs):
        super().__init__()
        input_dim = kwargs['input_dim']
        block_dim = kwargs['block_dim']
        output_dim = kwargs['output_dim']
        num_blocks = kwargs['num_blocks']
        layer_norm = kwargs['layer_norm']
        self.linear1 = LinearCurve(fix_nodes, input_dim, block_dim)
        self.linear2 = LinearCurve(fix_nodes, block_dim, output_dim)
        self.blocks = nn.ModuleList([ResidualBlockCurve(fix_nodes, **kwargs) for _ in range(num_blocks)])
        if layer_norm:
            self.norm = LayerNormCurve(fix_nodes, (block_dim,))
        else:
            self.register_buffer('norm', None)
  
    def forward(self, inputs, coeffs):
        outputs = self.linear1(inputs, coeffs)
        if self.norm:
            outputs = self.norm(outputs, coeffs)
        for block in self.blocks:
            outputs = block(outputs, coeffs)
        outputs = self.linear2(outputs, coeffs)
        return outputs

class RepresentationCurve(nn.Module):
    def __init__(self, num_nodes, fix_start, fix_end, step_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.fix_nodes = [fix_start] + [False] * (num_nodes - 2) + [fix_end]
        self.coeff = BezierCoefficient(num_nodes)
        self.step_size = step_size

    def import_parameters(self, model, index):
        parameters = list(self.parameters())[index::self.num_nodes]
        for param, base_param in zip(parameters, model.parameters()):
            param.data.copy_(base_param.data)

    def export_parameters(self, model, index):
        parameters = list(self.parameters())[index::self.num_nodes]
        for param, base_param in zip(parameters, model.parameters()):
            base_param.data.copy_(param.data)

    def export_parameters_t(self, model, t):
        coeffs = self.coeff(t)
        parameters = list(self.parameters())
        base_parameters = list(model.parameters())
        for param_index in range(len(base_parameters)):
            param = parameters[param_index * self.num_nodes:(param_index + 1) * self.num_nodes]
            base_param = base_parameters[param_index]
            base_param.data.zero_()
            for index in range(self.num_nodes):
                base_param.data.add_(param[index].data * coeffs[index])

    def init_linear(self):
        parameters = list(self.parameters())
        for param_index in range(len(parameters) // self.num_nodes):
            param = parameters[param_index * self.num_nodes:(param_index + 1) * self.num_nodes]
            for index in range(1, self.num_nodes - 1):
                alpha = index / (self.num_nodes - 1)
                param[index].data.copy_(alpha * param[-1].data + (1 - alpha) * param[0].data)
    
    def forward(self, inputs, t):
        pass

    def l1_penalty(self):
        penalty = 0
        for param in self.parameters():
            penalty += torch.abs(param).sum()
        return penalty

    @torch.no_grad()
    def sparsity(self):
        num_weights = 0
        num_zeros = 0
        for param in self.parameters():
            num_weights += param.numel()
            num_zeros += (torch.abs(param) < self.step_size / 2).sum().item()
        return num_zeros / num_weights
    
    def encode(self):
        encoder = deepCABAC.Encoder()
        for param in self.parameters():
            param = param.data.cpu().numpy()
            encoder.encodeWeightsRD(param, 0.0, self.step_size, 0.0)
        stream = encoder.finish().tobytes()
        return stream

    def decode(self, stream):
        decoder = deepCABAC.Decoder()
        decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
        for param in self.parameters():
            decoded_param = torch.tensor(decoder.decodeWeights(), device=param.device)
            param.data.copy_(decoded_param)
        decoder.finish()

class RepresentationCurve3D(RepresentationCurve):
    def __init__(self, num_nodes, fix_start, fix_end, **kwargs):
        step_size = kwargs['step_size']
        num_freqs = kwargs['num_freqs']
        super().__init__(num_nodes, fix_start, fix_end, step_size)
        input_dim = (2 * num_freqs + 1) * 3
        if num_freqs > 0:
            self.encoding = PositionalEncoding(3, num_freqs)
        else:
            self.encoding = nn.Identity()
        self.net = ResidualNetCurve(self.fix_nodes, input_dim=input_dim, **kwargs)

    def forward(self, inputs, t):
        outputs = self.encoding(inputs)
        outputs = self.net(outputs, self.coeff(t))
        outputs = torch.sigmoid(outputs)
        return outputs

def get_model(component, configs, device, curve=False, start_model=None, end_model=None):
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
    num_nodes = configs['num_nodes']
    fix_start = configs['fix_start']
    fix_end = configs['fix_end']
    if curve:
        model = RepresentationCurve3D(num_nodes, fix_start, fix_end, **kwargs).to(device)
        if fix_start and start_model:
            model.import_parameters(start_model, index=0)
        if fix_end and end_model:
            model.import_parameters(end_model, index=num_nodes - 1)
        if fix_start and fix_end and start_model and end_model:
            model.init_linear()
    else:
        model = Representation3D(**kwargs).to(device)
    return model