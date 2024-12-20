import pytorch_lightning as pl
import torch
from torch import nn
from my_custom_loss import CustomLoss
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset, WeightedRandomSampler
from data_module import RFDataset, load_fold_indices
import wandb
import numpy as np
import h5py
import os
import monai
from ConvTasNet import ConvTasNet
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from model_wavenet import WaveNet_separator, WaveNet_plus
from bit_estimator_models import EfficientNetDemod

from other_models import Autoencoder, FFCResnetBlock, Autoencoder_Bottlenecked
# from ffc_modules import FFCResNetBlock
import sys
sys.path.append('/dir')
from test_online import tester_loop
# import factorizer as ft
from omegaconf import OmegaConf

#import logging
#logging.basicConfig(filename='dataset_log.log', level=logging.INFO)

class VersaNetDemodulator(nn.Module):
    def __init__(self, config, config_path=None,  samples_per_symbol=16, num_symbols=None):
        super(VersaNetDemodulator, self).__init__()
        # Initialize VersaNet with the provided configuration
        self.versanet = VersaNet(config, config_path=config_path, skip_type=config.skip_type)
        # Initialize QPSKDemod with the provided parameters
        self.qpskdemod = QPSKDemod(samples_per_symbol=samples_per_symbol, num_symbols=num_symbols)
        self.transform_input = get_transform_function(config, mode='direct')
        self.transform_output = get_transform_function(config, mode='inverse')

    def forward(self, x):
        x = self.transform_input(x)
        # Pass the input through VersaNet
        versa_output = self.versanet(x)
        x = self.transform_output(versa_output['soi'])
        # The output of VersaNet is directly fed into QPSKDemod
        return x
        # return {'soi': x, 'bits': self.qpskdemod(x)} if self.training else {'soi': x}





class QUNet(nn.Module):
    def __init__(self, config, config_path=None,  samples_per_symbol=16, num_symbols=None):
        super(QUNet, self).__init__()
        quantize = config.quantize
        self.quant = torch.quantization.QuantStub() if quantize else nn.Identity()
        self.dequant = torch.quantization.DeQuantStub() if quantize else nn.Identity()
        skip_type = config.skip_type
        encoder_filters = config.encoder_filters

        
        self.encoder1 = CustomQuantizedEncoder(config.input_channels, encoder_filters[0], 1, quantize)
        self.encoder2 = CustomQuantizedEncoder(encoder_filters[0], encoder_filters[1], 2, quantize)
        self.encoder3 = CustomQuantizedEncoder(encoder_filters[1], encoder_filters[2], 2, quantize)
        self.decoder1 = CustomQuantizedDecoder(encoder_filters[2], encoder_filters[1], 2, quantize, skip_type)
        self.decoder2 = CustomQuantizedDecoder(encoder_filters[1], encoder_filters[0], 2, quantize, skip_type)
        self.pointwise = nn.Conv1d(encoder_filters[0], config.output_channels, kernel_size=1)
        self.qpskdemod = QPSKDemod(samples_per_symbol=samples_per_symbol, num_symbols=num_symbols)

    def forward(self, x):
        in0 = self.quant(x)
        en1 = self.encoder1(in0)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        de1 = self.decoder1(en3, en2)
        de2 = self.decoder2(de1, en1)
        out0 = self.pointwise(de2)
        out = self.dequant(out0)
        return {'soi': out, 'bits': self.qpskdemod(out)} if self.training else {'soi': out}
class CustomQuantizedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize):
        super(CustomQuantizedEncoder, self).__init__()
        self.per_channel_qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.per_tensor_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
        self.out_channels=out_channels
        conv1D =  nn.Conv1d
        self.conv1 = conv1D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = conv1D(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        # self.apply_qconfig(quantize)
        self.GN=nn.GroupNorm(4, self.out_channels)

    def apply_qconfig(self, quantize):
        if quantize:
            self.conv1.qconfig = self.per_tensor_qconfig
            self.conv2.qconfig = self.per_tensor_qconfig
    def forward(self, x):
        x = self.conv1(x)
        x = self.GN(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.GN(x)
        x = self.relu2(x)
        return x

class CustomQuantizedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize, skip_type):
        super(CustomQuantizedDecoder, self).__init__()
        
        self.per_channel_qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.per_tensor_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        )
        self.out_channels=out_channels
        
        conv1D = nn.Conv1d
        conv1DT = nn.ConvTranspose1d
        
        # Define skip_handler based on skip_type
        if skip_type == 'add':
            self.skip_handler = lambda x, skip: x + skip
            skip_coeff=1
        elif skip_type == 'concat':
            self.skip_handler = lambda x, skip: torch.cat((x, skip), dim=1)
            skip_coeff=2
        else:
            raise ValueError("Unsupported skip_type: " + skip_type)
            
        self.convT = conv1DT(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=-1+stride)
        self.relu1 = nn.LeakyReLU()
        self.conv = conv1D(skip_coeff*out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        # self.apply_qconfig(quantize)
        self.GN=nn.GroupNorm(4, self.out_channels)

    def apply_qconfig(self, quantize):
        if quantize:
            self.convT.qconfig = self.per_tensor_qconfig
            self.conv.qconfig = self.per_tensor_qconfig
    def forward(self, x, skip):
        x = self.convT(x)
        x = self.GN(x)
        x = self.relu1(x)
        x = self.skip_handler(x, skip)
        x = self.conv(x)
        x = self.GN(x)
        x = self.relu2(x)
        return x        

class Scriptable_QuantizedEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize):
        super(Scriptable_QuantizedEncoder, self).__init__()
        self.out_channels = out_channels

        # Convolution layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.GN1 = nn.GroupNorm(4, self.out_channels)
        self.GN2 = nn.GroupNorm(4, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.GN1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.GN2(x)
        x = self.relu2(x)
        return x
 
class Scriptable_QuantizedEncoder_depth(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize):
        super(Scriptable_QuantizedEncoder_depth, self).__init__()
        self.out_channels = out_channels

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # Pointwise Convolution
        self.relu1 = nn.LeakyReLU()
        
        # Depthwise Convolution
        self.depthwise_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.pointwise_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)  # Pointwise Convolution
        self.relu2 = nn.LeakyReLU()
        
        self.GN = nn.GroupNorm(4, self.out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.GN(x)
        x = self.relu1(x)

        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.GN(x)
        x = self.relu2(x)

        return x       
        
class Scriptable_QuantizedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize):
        super(Scriptable_QuantizedDecoder, self).__init__()
        
        self.out_channels = out_channels
        
        # Using Conv1d and ConvTranspose1d layers
        self.convT = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1)
        self.relu1 = nn.LeakyReLU()
        self.conv = nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.GN = nn.GroupNorm(4, out_channels)

    def forward(self, x, skip):
        x = self.convT(x)
        x = self.GN(x)
        x = self.relu1(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        x = self.GN(x)
        x = self.relu2(x)
        return x

class Scriptable_QuantizedDecoder_depth(nn.Module):
    def __init__(self, in_channels, out_channels, stride, quantize):
        super(Scriptable_QuantizedDecoder_depth, self).__init__()
        
        self.out_channels = out_channels
        
        # Depthwise and Pointwise ConvTranspose
        self.depthwise_convT = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, groups=in_channels)
        self.pointwise_convT = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.LeakyReLU()

        # Adjust the number of input channels for the depthwise convolution, considering the concatenated skip connections
        combined_channels = out_channels + out_channels  # Adjust as needed based on the skip connection structure

        # Depthwise and Pointwise Convolution
        self.depthwise_conv = nn.Conv1d(combined_channels, combined_channels, kernel_size=3, padding=1, groups=combined_channels)
        self.pointwise_conv = nn.Conv1d(combined_channels, out_channels, kernel_size=1)
        self.relu2 = nn.LeakyReLU()

        self.GN = nn.GroupNorm(4, out_channels)

    def forward(self, x, skip):
        x = self.depthwise_convT(x)
        x = self.pointwise_convT(x)
        x = self.GN(x)
        x = self.relu1(x)

        x = torch.cat((x, skip), dim=1)
        
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.GN(x)
        x = self.relu2(x)
        return x

class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()
        # Define an identity layer
        self.Identity_out = nn.Identity()

    def forward(self, x):
        # The output is exactly the same as the input
        return self.Identity_out(x)
        
class Scriptable_QUNet(nn.Module):
    def __init__(self, config, config_path=None,  samples_per_symbol=16, num_symbols=None):
        super(Scriptable_QUNet, self).__init__()
        quantize = config.quantize
        self.quant = torch.quantization.QuantStub() if quantize else nn.Identity()
        self.dequant = torch.quantization.DeQuantStub() if quantize else nn.Identity()
        encoder_filters = config.encoder_filters
        self.ch2batch = getattr(config, 'ch2batch', False)
        conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1 
        self.config=config

        if config.channel_wise:
            self.encoder1 = Scriptable_QuantizedEncoder_depth(config.input_channels, encoder_filters[0], 1, quantize)
            self.encoder2 = Scriptable_QuantizedEncoder_depth(encoder_filters[0], encoder_filters[1], 2, quantize)
            self.encoder3 = Scriptable_QuantizedEncoder_depth(encoder_filters[1], encoder_filters[2], 2, quantize)
        else:
            self.encoder1 = Scriptable_QuantizedEncoder(config.input_channels, encoder_filters[0], 1, quantize)
            self.encoder2 = Scriptable_QuantizedEncoder(encoder_filters[0], encoder_filters[1], 2, quantize)
            self.encoder3 = Scriptable_QuantizedEncoder(encoder_filters[1], encoder_filters[2], 2, quantize)
            
        if config.get('decoder_channel_wise', False):
            self.decoder1 = Scriptable_QuantizedDecoder_depth(encoder_filters[2], encoder_filters[1], 2, quantize)
            self.decoder2 = Scriptable_QuantizedDecoder_depth(encoder_filters[1], encoder_filters[0], 2, quantize)
        else:
            self.decoder1 = Scriptable_QuantizedDecoder(encoder_filters[2], encoder_filters[1], 2, quantize)
            self.decoder2 = Scriptable_QuantizedDecoder(encoder_filters[1], encoder_filters[0], 2, quantize)
            
        self.pointwise = nn.Conv1d(encoder_filters[0], config.output_channels, kernel_size=1)
        if config.bottleneck_type=='lstm':
            self.bottleneck = LSTMBottleneck(encoder_filters[2], config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim)
        else:
            self.bottleneck = IdentityModule()

    def forward(self, x):
        in0 = self.quant(x)
        # if self.ch2batch:
        #     batch, channels, length = in0.shape
        #     in0 = in0.view(batch * channels, 1, length)
        en1 = self.encoder1(in0)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        bn = self.bottleneck(en3)
        de1 = self.decoder1(bn, en2)
        de2 = self.decoder2(de1, en1)
        out0 = self.pointwise(de2)
        # if self.ch2batch:
        #     out0 = out0.view(batch, channels, -1)
        out = self.dequant(out0)
        return out



class Scriptable_QUNet2(nn.Module):
    def __init__(self, config, config_path=None, samples_per_symbol=16, num_symbols=None):
        super(Scriptable_QUNet2, self).__init__()
        quantize = config.quantize
        self.quant = torch.quantization.QuantStub() if quantize else nn.Identity()
        self.dequant = torch.quantization.DeQuantStub() if quantize else nn.Identity()
        encoder_filters = config.encoder_filters
        encoder_strides = config.encoder_strides[:len(encoder_filters)]
        self.ch2batch = getattr(config, 'ch2batch', False)
        conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1
        self.config = config
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Create encoder modules
        in_channels = config.input_channels
        for out_channels, stride in zip(encoder_filters, encoder_strides):
            if config.channel_wise:
                encoder = Scriptable_QuantizedEncoder_depth(in_channels, out_channels, stride, quantize)
            else:
                encoder = Scriptable_QuantizedEncoder(in_channels, out_channels, stride, quantize)
            self.encoders.append(encoder)
            in_channels = out_channels

        # Create decoder modules
        decoder_strides = encoder_strides[1:][::-1]  # Reverse and exclude the first stride
        for i in range(len(encoder_filters) - 1):
            in_channels = encoder_filters[-(i+1)]
            out_channels = encoder_filters[-(i+2)]
            stride = decoder_strides[i]
            if config.get('decoder_channel_wise', False):
                decoder = Scriptable_QuantizedDecoder_depth(in_channels, out_channels, stride, quantize)
            else:
                decoder = Scriptable_QuantizedDecoder(in_channels, out_channels, stride, quantize)
            self.decoders.append(decoder)

        self.pointwise = nn.Conv1d(encoder_filters[0], config.output_channels, kernel_size=1)
        if config.bottleneck_type == 'lstm':
            self.bottleneck = LSTMBottleneck(encoder_filters[-1], config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim)
        else:
            self.bottleneck = IdentityModule()

    def forward(self, x):
        in0 = self.quant(x)
        encoder_outputs = [in0]

        # Forward pass through encoders
        for encoder in self.encoders:
            out = encoder(encoder_outputs[-1])
            encoder_outputs.append(out)

        bn = self.bottleneck(encoder_outputs[-1])

        # Forward pass through decoders
        out = bn
        for i, decoder in enumerate(self.decoders):
            out = decoder(out, encoder_outputs[-(i+2)])

        out0 = self.pointwise(out)
        output = self.dequant(out0)
        return output




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        if stride>0:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else None
        else:
            self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=-1*stride, padding=1)
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=-1*stride) if in_channels != out_channels else None
            
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, config, config_path=None, samples_per_symbol=16, num_symbols=None):
        super(ResNet, self).__init__()
        self.in_channels = config.input_channels
        layers = []

        # Encoder: Increasing channels and including strides
        for channels, stride in zip(config.encoder_filters, reversed(config.encoder_strides[:len(config.encoder_filters)])):
            layers.append(ResidualBlock(self.in_channels, channels, stride))
            self.in_channels = channels

        # Assuming strides are not required for decoder, or they are constant (usually stride=1 in decoding paths)
        for channels,stride in zip(reversed(config.encoder_filters[:-1]),config.encoder_strides[:len(config.encoder_filters)]):
            layers.append(ResidualBlock(self.in_channels, channels, -1*stride))
            self.in_channels = channels

        # Final layer to match output_channels, usually stride=1
        layers.append(ResidualBlock(self.in_channels, config.output_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTasNet_Demodulator(nn.Module):
    def __init__(self, config, config_path=None, samples_per_symbol=16, num_symbols=None):
        super(ConvTasNet_Demodulator, self).__init__()

        self.conv_tas_net = ConvTasNet(config.ConvTasNet_N, config.ConvTasNet_L, config.ConvTasNet_B, config.ConvTasNet_H, config.ConvTasNet_P, config.ConvTasNet_X, config.ConvTasNet_R)

        self.qpskdemod = QPSKDemod(samples_per_symbol=samples_per_symbol, num_symbols=num_symbols)

    def forward(self, x):
        x = self.conv_tas_net(x)
        return {'soi': x, 'bits': self.qpskdemod(x)} if self.training else {'soi': x}

class LaMaResAutoEncoder(nn.Module):
    def __init__(self, config, config_path=None, samples_per_symbol=16, num_symbols=None):
        super(LaMaResAutoEncoder, self).__init__()
        
        # Example instantiation
        # lama_filters = [4,                 64,128,256,512,1024,        512,256,128,64]
        # kernel_sizes = [3,            3,3,3,3,                     2,2,2,2]
        # strides = [1,                 2,2,2,2,                     -2,-2,-2,-2]
        # ratio_gins = [0.5,           .5,.5,0.5, .5,               .5,.5,0.5,.5]
        # ratio_gouts = [0.5,          .5,.5,0.5, .5,               .5,0.5,.5,0.125]
        # output_channels=2
        # Define DynUNet
        self.Lama = Autoencoder(config.lama_filters, config.lama_kernel_sizes, config.lama_strides, config.lama_ratio_gins, config.lama_ratio_gouts, config.output_channels, conv_dim=1, verbose=False)
        self.qpskdemod = QPSKDemod(samples_per_symbol=samples_per_symbol, num_symbols=num_symbols)

    def forward(self, x):
        x = self.Lama(x)
        return {'soi': x, 'bits': self.qpskdemod(x)} if self.training else {'soi': x}


from torch.quantization import FakeQuantize
import torch
import torch.nn as nn

from torch.quantization import get_default_qat_qconfig, QConfig
from torch.quantization.observer import (
    MinMaxObserver, PerChannelMinMaxObserver, 
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
)

def apply_quant_config(module, cfg):
    def apply_quant(module):
        # Function to convert observer string to the actual observer class
        def get_observer(observer_str):
            observer_map = {
                "MinMaxObserver": MinMaxObserver,
                "PerChannelMinMaxObserver": PerChannelMinMaxObserver,
                "MovingAverageMinMaxObserver": MovingAverageMinMaxObserver,
                "MovingAveragePerChannelMinMaxObserver": MovingAveragePerChannelMinMaxObserver
            }
            return observer_map.get(observer_str, MinMaxObserver)

        # Function to convert qscheme string to torch enum
        def get_qscheme(qscheme_str):
            qscheme_map = {
                "per_tensor_affine": torch.per_tensor_affine,
                "per_channel_affine": torch.per_channel_affine,
                "per_tensor_symmetric": torch.per_tensor_symmetric,
                "per_channel_symmetric": torch.per_channel_symmetric
            }
            return qscheme_map.get(qscheme_str, torch.per_tensor_symmetric)

        # Convert dtype string to torch dtype
        def get_dtype(dtype_str):
            dtype_map = {
                "quint8": torch.quint8,
                "qint8": torch.qint8,
                "int32": torch.int32
            }
            return dtype_map.get(dtype_str, torch.quint8)

        if isinstance(module, nn.ConvTranspose1d):
            observer = get_observer(cfg.ConvTranspose1d_observer)
            qscheme = get_qscheme(cfg.ConvTranspose1d_qscheme)
            module.qconfig = QConfig(
                activation=observer.with_args(dtype=get_dtype(cfg.activation_dtype)),
                weight=observer.with_args(dtype=get_dtype(cfg.weight_dtype), qscheme=qscheme)
            )
        elif isinstance(module, nn.Conv1d):
            observer = get_observer(cfg.Conv1d_observer)
            qscheme = get_qscheme(cfg.Conv1d_qscheme)
            module.qconfig = QConfig(
                activation=observer.with_args(dtype=get_dtype(cfg.activation_dtype)),
                weight=observer.with_args(dtype=get_dtype(cfg.weight_dtype), qscheme=qscheme)
            )
        elif isinstance(module, (nn.LSTM, nn.Linear)):  # Add layers to exclude
            module.qconfig = None  # This will skip quantization for these layers
        else:
            module.qconfig = get_default_qat_qconfig('fbgemm')

        for child in module.children():
            apply_quant(child)

        module.train()
        torch.quantization.prepare_qat(module, inplace=True)

    apply_quant(module)


def print_qconfig(module, prefix=""):
    for name, child in module.named_children():
        if hasattr(child, 'qconfig') and child.qconfig is not None:
            print(f"Layer: {prefix + name}, QConfig: {child.qconfig}")
        print_qconfig(child, prefix + name + ".")



model_selector = {
            "VersaNetDemodulator": VersaNetDemodulator,
            "VersaNetDemodMod": VersaNetDemodMod,
            "WaveNet_separator": WaveNet_separator,
            "WaveNet_DemodMod": WaveNet_DemodMod,
            "DynUnet": DynUnet,
            "CustomEfficientNetModule": CustomEfficientNetModule,
            "DynUnet_DemodMod":DynUnet_DemodMod,
            "DynUNet_time_freq":DynUNet_time_freq,
            "DynUNet_time_freq_Hankel":DynUNet_time_freq_Hankel,
            # "Deconver":Deconver,
            "DynUNet_time_freq_DemodMod":DynUNet_time_freq_DemodMod,
            "DynUNet_time_freq_CP":DynUNet_time_freq_CP,
            "UNet2d_Hankel_STFT":UNet2d_Hankel_STFT,
            "SwinUNETR":SwinUNETR,
            "SwinUNETR_Demodulator":SwinUNETR_Demodulator,
            "unetpp":unetpp,
            "unetpp2d":unetpp2d,
            "LaMaResAutoEncoder":LaMaResAutoEncoder,
            "LaMaResAutoEncoder_Bottlenecked":LaMaResAutoEncoder_Bottlenecked,
            "ResidualAutoencoder":ResidualAutoencoder,
            "WaveNet_plus":WaveNet_plus,
            "EfficientNetDemod":EfficientNetDemod,
            "CombinedDemodulator":CombinedDemodulator,
            "VersaNetDemodulator_Mask":VersaNetDemodulator_Mask,
            "VersaNetDemodulator_Res":VersaNetDemodulator_Res,
            "VersaNetDemodulator_BottleNeck_Res":VersaNetDemodulator_BottleNeck_Res,
            "VersaNetDemodulator_BottleNeck_Mask":VersaNetDemodulator_BottleNeck_Mask,
            "WNetDemodulator":WNetDemodulator,
            "WNetDemodulator_diff":WNetDemodulator_diff,
            "ConvTasNet_Demodulator":ConvTasNet_Demodulator,
            "QUNet":QUNet,
            "ResNet":ResNet,
            "Scriptable_QUNet":Scriptable_QUNet,
            "Scriptable_QUNet2":Scriptable_QUNet2,
            "DeepDemodulator":DeepDemodulator,
            "EfficientNetDemodulator":EfficientNetDemodulator,
            "Scriptable_QUNet_DeepSup":Scriptable_QUNet_DeepSup,
            "UnetMonaiNetDemodulator":UnetMonaiNetDemodulator,
            "AdaptiveMultiExitUNet":AdaptiveMultiExitUNet,
            "QPSKBitEstimator":QPSKBitEstimator,
            "MultiExitUNet":MultiExitUNet,#,
            # "FactorizerAsh":FactorizerAsh
}





def attach_output_hooks(model, layers_dict):
    outputs = {}

    for layer_id, info in layers_dict.items():
        def hook(module, input, output, name=layer_id):
            outputs[name] = output.detach()

        layer_name = info["name"]
        if layer_name in dict(model.named_modules()):
            layer = dict(model.named_modules())[layer_name]
            layer.register_forward_hook(hook)

    return outputs
    
def compute_attention_map(x):
    # Sum of squares across the channel dimension
    attention_map = torch.sum(x ** 2, dim=1, keepdim=True)

    # Normalize by the number of channels
    num_channels = x.shape[1]
    normalized_attention_map = attention_map / num_channels
    # print(x.shape, normalized_attention_map.shape)

    return normalized_attention_map
def distillation_loss(student_outputs, teacher_outputs, student_layers, teacher_layers, criterion=nn.MSELoss()):
    total_loss = 0.0

    for layer_id in student_layers:
        student_layer_output = student_outputs[layer_id]
        teacher_layer_output = teacher_outputs[layer_id]

        if student_layer_output.shape == teacher_layer_output.shape:
            # Calculate MSE directly if the shapes are the same
            dum01 = criterion(student_layer_output, teacher_layer_output)
            layer_loss = 10 * torch.log10(dum01+1e-5)
        else:
            # Compute attention maps
            student_attention = compute_attention_map(student_layer_output)
            teacher_attention = compute_attention_map(teacher_layer_output)
            # Calculate loss based on attention maps
            dum01 = criterion(student_attention, teacher_attention)
            layer_loss = 10 * torch.log10(dum01+1e-5)
        # print(layer_loss)
        total_loss += student_layers[layer_id]["weight"] * layer_loss

    return total_loss


class LitModel(pl.LightningModule):
    def __init__(self, config, config_path=None):
        super().__init__()
        # Save the entire config as model hyperparameters
        self.save_hyperparameters(config)
        print(f'self.hparams={self.hparams}')
        # Now the entire config dictionary is accessible through self.hparams
        # Instantiate the selected model class
        self.student = model_selector[self.hparams.model.model_type](self.hparams.model, config_path=config_path)
        # self.distillation()
        if self.hparams.model.pre_trained_checkpoint:
            self.load_pretrained_checkpoint()
        if self.hparams.model.quantize:
            apply_quant_config(self.student, self.hparams.quantization)
            # print(f'printing implemented quantization config')
            # print_qconfig(self.student)
            # print(f'printing requested quantization config')
            # print(self.hparams.quantization)
            print(f'printing model')
            print(self.student)
        self.criterion = CustomLoss(loss_type=self.hparams.trainer.backward_option, combined_loss_ratio=self.hparams.trainer.combined_loss_ratio, l1_weight=self.hparams.trainer.l1_weight, l2_weight=self.hparams.trainer.l2_weight)

    

    def forward(self, x):
        return self.student(x)
#

    def distillation(self):
        dum0=self.hparams.distillation.teacher_timestamp
        formatted_timestamp = str(dum0)[:8]  + str(dum0)[8:]
        teacher_config_path = f'/dir/lightning_logs/{formatted_timestamp}/config.yaml'
        teacher_config = OmegaConf.load(teacher_config_path) 
        self.teacher = model_selector[teacher_config.model.model_type](teacher_config.model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Path to your YAML file
        yaml_file_path = '/dir/Distillation_config.yaml'
        
        # Load the configuration
        config = OmegaConf.load(yaml_file_path)
        
        # Assuming the YAML file has specific fields for student and teacher target layers
        self.student_target_layers = config.student_target_layers
        self.teacher_target_layers = config.teacher_target_layers

        self.student_outputs = attach_output_hooks(self.student, self.student_target_layers)
        self.teacher_outputs = attach_output_hooks(self.teacher, self.teacher_target_layers)
    

    

    


    

    def training_step(self, batch, batch_idx):
        mixture, soi, bits, target_sinr, _, interference_type = batch
        intermediates=None
        if self.hparams.trainer.Which_Target == "bit":
            model_input = bits
        elif self.hparams.trainer.Which_Target == "soi":
            model_input = mixture
            
        if self.hparams.trainer.IsDeepSupervision:
            output, intermediates = self(model_input)
        else:
            output = self(model_input)
            
        if self.hparams.distillation.Is_distillation:
            loss, mse_db = self.criterion(output, {'soi': soi, 'bits': bits, 'mixture':mixture}, interference_sig_type=interference_type, model_input=model_input, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, model_parameters=self.student, DeepSup=intermediates, hparam=self.hparams)
            
            with torch.no_grad():
                teacher_output = self.teacher(mixture)
            student_output = self.student(mixture)
            distillation_loss_value = distillation_loss(self.student_outputs, self.teacher_outputs, self.student_target_layers, self.teacher_target_layers)
            lambda_distillation = self.hparams.distillation.lambda_value
            total_loss = loss + lambda_distillation * distillation_loss_value
            return total_loss
        else:
            
            # Calculate loss
            if self.hparams.trainer.backward_option in ['ber_loss', 'combined_ber_soi']:
                loss, bit_error_rate = self.criterion(output, {'soi': soi, 'bits': bits, 'mixture':mixture}, interference_sig_type=interference_type, model_input=model_input, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, model_parameters=self.student, DeepSup=intermediates, hparam=self.hparams)
            else:
                loss, mse_db = self.criterion(output, {'soi': soi, 'bits': bits, 'mixture':mixture}, interference_sig_type=interference_type, model_input=model_input, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, model_parameters=self.student, DeepSup=intermediates, hparam=self.hparams)
        
        # Log metrics
        if self.trainer.global_rank == 0 and self.current_epoch % self.hparams.trainer.N_log == 0 and batch_idx == 0:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            if self.hparams.trainer.backward_option in ['ber_loss', 'combined_ber_soi']:
                wandb.log({"train/loss": loss, "train/bit_error_rate": bit_error_rate, "lr": lr, "current_epoch": self.current_epoch})
            else:
                wandb.log({"train/loss": loss, "train/mse_db": mse_db, "lr": lr, "current_epoch": self.current_epoch})
            # self.log('learning_rate', lr, on_step=True, logger=True)
    
        # Check if separate optimizers are to be used
        if self.hparams.trainer.Is_separate_param:
            l0_norm_group2 = compute_avg_l0_norm(self.group2_params)
            wandb.log({"train/l0_norm_gates": l0_norm_group2}) 
            # Manually handle optimization
            opt1, opt2 = self.optimizers()
            opt1.zero_grad()
            opt2.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            opt2.step()
        else:
            # Automatic optimization handled by PyTorch Lightning
            return loss

    def validation_step(self, batch, batch_idx):

        if self.trainer.global_rank == 0:
            mixture, soi, bits, target_sinr, _, interference_type = batch
            intermediates=None
            if self.hparams.trainer.Which_Target == "bit":
                model_input = bits
            elif self.hparams.trainer.Which_Target == "soi":
                model_input = mixture
            if self.hparams.trainer.IsDeepSupervision:
                output, intermediates = self(model_input)
                IsDeepSup=intermediates
            else:
                output = self(model_input)
            
            if self.hparams.trainer.backward_option in ['ber_loss', 'combined_ber_soi']:
                val_loss, bit_error_rate = self.criterion(output, {'soi': soi, 'bits': bits}, interference_sig_type=interference_type, model_input=model_input, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, model_parameters=self.student, DeepSup=intermediates, hparam=self.hparams)
            else:
                val_loss, mse_db = self.criterion(output, {'soi': soi, 'bits': bits}, interference_sig_type=interference_type, model_input=model_input, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, model_parameters=self.student, DeepSup=intermediates, hparam=self.hparams)
            if self.trainer.global_rank == 0:#  and batch_idx % self.hparams.trainer.N_log == 0:
                #self.log('val_loss', loss_dict['loss'], on_epoch=True, prog_bar=True, logger=True)
                if self.hparams.trainer.backward_option in ['ber_loss', 'combined_ber_soi']:
                    wandb.log({"val/loss": val_loss, "val/bit_error_rate": bit_error_rate, "current_epoch": self.current_epoch})
                    
                else:
                    wandb.log({"val/loss": val_loss, "val/mse_db": mse_db, "current_epoch": self.current_epoch})
                
            return val_loss
        else:
            return None
  

        
    def on_train_start(self):
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in self.student.parameters())
        total_params_million = total_params / 1_000_000.0
        # Format to 2 decimal places
        total_params_million_formatted = round(total_params_million, 4)

        # Log the number of parameters to wandb
        wandb.log({'total_model_parameters': total_params_million_formatted})
        
    def separate_parameters(self):
        group1_params = []
        group2_params = []
        gate_indices = [1, 6]
    
        # Add the specified parameters to group2
        for i in range(len(self.student.versanet.encoders)):
            for idx in gate_indices:
                group2_params.extend(self.student.versanet.encoders[i].conv[idx].parameters())
                if i < len(self.student.versanet.encoders) - 1:
                    group2_params.extend(self.student.versanet.decoders[i].conv_block[idx].parameters())
        group2_param_ids = {id(p) for p in group2_params}
    
        # Add the rest of the parameters to group1
        for param in self.student.parameters():
            if id(param) not in group2_param_ids:
                group1_params.append(param)
    
        return group1_params, group2_params
    
        
    def configure_optimizers(self):
        optim_params = self.hparams.optimizer
        sched_params = self.hparams.scheduler

        if self.hparams.trainer.Is_separate_param:
            self.automatic_optimization = False
            self.group1_params, self.group2_params = self.separate_parameters()
            optimizer1 = torch.optim.Adam(self.parameters(), lr=optim_params['lr0'])
            optimizer2 = torch.optim.Adam(self.parameters(), lr=optim_params['lr0_2'])
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=self.hparams.trainer.max_epochs, eta_min=0)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=self.hparams.trainer.max_epochs, eta_min=0)
            return [
            {'optimizer': optimizer1, 'lr_scheduler': scheduler1},
            {'optimizer': optimizer2, 'lr_scheduler': scheduler2}
        ]
            
        else:
            # Select optimizer
            if optim_params['optimizer_type'] == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=optim_params['lr0'])
                if self.hparams.model.model_type=='CombinedDemodulator':
                    optimizer = torch.optim.Adam(self.student.efficientnet_demod.parameters(), lr=optim_params['lr0'])
                    
            elif optim_params['optimizer_type'] == 'adamw':
                optimizer = torch.optim.AdamW(self.parameters(), lr=optim_params['lr0'])
            elif optim_params['optimizer_type'] == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=optim_params['lr0'], momentum=optim_params.get('momentum', 0.9))
    
            # Scheduler selection
            if self.hparams.scheduler['scheduler_type'] == 'cosine_annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.trainer.max_epochs, eta_min=0)
            elif self.hparams.scheduler['scheduler_type'] == 'decay':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.scheduler.step_size, gamma=self.hparams.scheduler.gamma)
            elif self.hparams.scheduler['scheduler_type'] == 'fix':
                scheduler = None  # If scheduler is fixed, we don't use a scheduler
        
            # Scheduler configuration
            scheduler_config = {'scheduler': scheduler, 'monitor': 'val_loss'} if scheduler else None
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config
            } if scheduler_config else optimizer



    def on_train_epoch_end(self, unused=None):
        # Check if the current epoch is one where we should save the model
        if self.trainer.global_rank == 0 and self.current_epoch > 0:
            # If you want to access the global step (total iterations since the beginning of training)
            # global_step = self.trainer.global_step
            # print(f"current epoch: {self.current_epoch}, Global Step: {global_step}")
            if self.current_epoch == 1 or self.current_epoch % self.hparams.trainer.N_save_model == 0:
                # Check if we should save the full model or just the state dictionary
                if self.hparams.trainer.get('Is_save_full_model', False):
                    # Save the entire model
                    model_path = os.path.join(self.hparams.trainer.model_save_dir, f'{self.hparams.timestamp}_full.pth')
                    torch.save(self.student, model_path)
                else:
                    # Save only the state dictionary
                    model_path = os.path.join(self.hparams.trainer.model_save_dir, f'{self.hparams.timestamp}.pt')
                    
                    torch.save(self.student.state_dict(), model_path)
            
        if self.trainer.global_rank == 0 and self.current_epoch>0 and (self.current_epoch ==1 or self.current_epoch % self.hparams.tester.each_N_epochs == 0):
            _, _, STDs, _, average_ber, STD_BERs, MSE_Score, BER_Score, _, _ = tester_loop(self.hparams.timestamp, self.hparams.tester.N_per_sinr_test, sinr_values=list(range(-30, 1, 3)), batch_size=1, Force_CPU=False,Is_inference=False, seed=42, eps_sinr=self.hparams.trainer.Increment_snr_epsilon, Is_increment=self.hparams.trainer.Is_increment, hparams = self.hparams)
            wandb.log({"test/mse_score": MSE_Score, "test/ber_score": BER_Score, "test/ber": np.mean(average_ber), "current_epoch": self.current_epoch})
        scheduler = self.lr_schedulers()


    def train_dataloader(self):
        # Create training dataset and weights
        if getattr(self.hparams.dataloader, 'Is_weighted_sampler', False):
            dataset, self.train_weights = self.create_dataset(is_train=True)
            # Create a WeightedRandomSampler
            weighted_sampler = WeightedRandomSampler(self.train_weights, len(self.train_weights))
    
            return DataLoader(dataset, 
                          batch_size=self.hparams.dataloader.batch_size, 
                          num_workers=self.hparams.dataloader.N_worker, 
                          pin_memory=True, 
                          sampler=weighted_sampler)
        else:
            dataset= self.create_dataset(is_train=True)
            train_sampler = DistributedSampler(dataset, shuffle=True)
            return DataLoader(dataset, 
                          batch_size=self.hparams.dataloader.batch_size, 
                          num_workers=self.hparams.dataloader.N_worker, 
                          pin_memory=True, 
                          sampler=train_sampler)

    def val_dataloader(self):
        val_batch_size = getattr(self.hparams.dataloader, 'val_batch_size', self.hparams.dataloader.batch_size)
        if getattr(self.hparams.dataloader, 'Is_weighted_sampler', False):
            print(f'weighted sampler is used!')
            dataset, self.val_weights = self.create_dataset(is_train=False)
            # Create a WeightedRandomSampler
            weighted_sampler = WeightedRandomSampler(self.val_weights, len(self.val_weights))
            return DataLoader(dataset, 
                          batch_size=val_batch_size, 
                          num_workers=self.hparams.dataloader.N_worker, 
                          pin_memory=True, 
                          sampler=weighted_sampler)
        else:
            print(f'Distributed sampler is used!')
            # Create validation dataset
            dataset = self.create_dataset(is_train=False)
    
            # DistributedSampler for validation dataset
            val_sampler = DistributedSampler(dataset, shuffle=False)
    
            return DataLoader(dataset, 
                              batch_size=val_batch_size, 
                              num_workers=self.hparams.dataloader.N_worker, 
                              pin_memory=True, 
                              sampler=val_sampler)
    def create_dataset(self, is_train):
        fold_number = self.hparams.trainer.fold
        soi_type = self.hparams.dataloader.SOI_type
    
        sinr_start = self.hparams.dataloader.sinr_start
        sinr_end = self.hparams.dataloader.sinr_end
        
        if self.hparams.dataloader.interference_idx[0]!= 'no':
            interference_types = [self.hparams.dataloader.All_interference_type[i] for i in self.hparams.dataloader.interference_idx]
        else:
            interference_types = self.hparams.dataloader.interference_type
        
        unseen_interference_type_idx = self.hparams.dataloader.unseen_interference_type_idx

        combined_dataset = []
    
        # for interference_type in interference_types:
        for idx, interference_type in enumerate(interference_types):
            if idx == unseen_interference_type_idx and is_train:
                continue
            else:
                with h5py.File(os.path.join(self.hparams.dataloader.train_dataset_dir, interference_type + '_raw_data.h5'), 'r') as data_h5file:
                    sig_data = np.array(data_h5file.get('dataset'))
        
                largest_start_idx = sig_data.shape[1] - self.hparams.dataloader.sig_len
                fold_indices = load_fold_indices(f'{self.hparams.dataloader.fold_indices_dir}{self.hparams.dataloader.N_total_folds}folds_{interference_type}_fold_assignment.json')
        
                if is_train and hasattr(self.hparams.dataloader, 'N_per_superframe'):
                    N_per_superframe = self.hparams.dataloader.N_per_superframe
                else:
                    N_per_superframe = 1
        
                dataset = RFDataset(sig_data, self.hparams.dataloader.sig_len, fold_indices, fold_number, soi_type, sinr_start, sinr_end, largest_start_idx, N_per_superframe, is_train, interference_type)
                combined_dataset.append(dataset)


            # Print the size of each individual dataset
            if is_train:
                print(f"Training Dataset for {interference_type} added with size: {len(dataset)}")
            else:
                print(f"Validation Dataset for {interference_type} added with size: {len(dataset)}")
        if getattr(self.hparams.dataloader, 'Is_weighted_sampler', False):
            weights = []
            for dataset in combined_dataset:
                dataset_weight = 1.0 / len(dataset)
                weights.extend([dataset_weight] * len(dataset))
            
            # Print the total size of the combined dataset and the first few weights
            self.steps_per_epoch = sum(len(d) for d in combined_dataset) // self.hparams.dataloader.batch_size
            print(f"Total size of combined dataset: {sum(len(d) for d in combined_dataset)}")

            return ConcatDataset(combined_dataset), weights
        else:
            # Print the total size of the combined dataset
            self.steps_per_epoch = sum(len(d) for d in combined_dataset) // self.hparams.dataloader.batch_size

            print(f"Total size of combined dataset: {sum(len(d) for d in combined_dataset)}")

    
            return ConcatDataset(combined_dataset)
        
    def load_pretrained_checkpoint(self, project_name="Solid"):
        # Check for direct checkpoint path first
        full_checkpoint_path = self.hparams.model.get('full_checkpoint_path')
        if full_checkpoint_path and os.path.exists(full_checkpoint_path):
            print(f"Loading pre-trained weights from {full_checkpoint_path}")
            self.student.load_state_dict(torch.load(full_checkpoint_path))
            return
        if self.hparams.model.Exits_timestamps:   
            # Iterate over each timestamp and load the corresponding model weights
            # for i, timestamp in enumerate(self.hparams.model.Exits_timestamps):
            # for i, (timestamp, Is_load, freeze) in enumerate(zip(self.hparams.model.Exits_timestamps, self.hparams.model.Exits_timestamps_preload_cfg, self.hparams.model.Exits_timestamps_freeze)):
            for i, (timestamp, freeze, preload_cfg) in enumerate(zip(self.hparams.model.Exits_timestamps, self.hparams.model.Exits_timestamps_freeze, self.hparams.model.Exits_timestamps_preload_cfg)):

                print(f'pretrain loading flag={preload_cfg}')
                print(f'freeze={freeze}')
                print(f'timestamp={timestamp}')
                if preload_cfg==1:
                    formatted_timestamp = str(timestamp)[:8] + '_' + str(timestamp)[8:]
                    checkpoint_path = f'/dir/models/{formatted_timestamp}.pt'
    
                    if os.path.exists(checkpoint_path):
                        print(f"Pre-trained weights for exit {i} from {checkpoint_path} loading...!")
                        model_checkpoint = torch.load(checkpoint_path)
                        self.student.unets[i].load_state_dict(model_checkpoint)
                        print(f"Pre-trained weights for exit {i} from {checkpoint_path} Loaded!")
                        if freeze == 1:
                            for param in self.student.unets[i].parameters():
                                param.requires_grad = False
                            print(f"Model exit {i} has been frozen.") 
                    else:
                        print(f"Warning: Checkpoint file for exit {i} at {checkpoint_path} not found. Continuing with uninitialized model for this exit.")
                 
        else:
            pre_trained_checkpoint = self.hparams.model.get('pre_trained_checkpoint')
            if pre_trained_checkpoint:
                formatted_timestamp = str(pre_trained_checkpoint)[:8] + '_' + str(pre_trained_checkpoint)[8:]
                checkpoint_path = f'/dir/models/{formatted_timestamp}.pt'
                if os.path.exists(checkpoint_path):
                    print(f"Loading pre-trained weights from {checkpoint_path}")
                    #checkpoint = torch.load(checkpoint_path)
                    self.student.load_state_dict(torch.load(checkpoint_path))
                else:
                    print(f"Warning: Checkpoint file {checkpoint_path} not found. Continuing with uninitialized model.")
    
        # You can also implement other methods such as on_train_batch_end, on_train_epoch_start, etc.

        
class VersaNet(nn.Module):
    def __init__(self, config, config_path=None, skip_type='concat'):
        super(VersaNet, self).__init__()
        self.config=config
        self.apply(lambda m: initialize_weights(m, config.weight_init_type, config.Is_weight_init))
        self.encoders = create_encoders(config)
        self.bottleneck = create_bottleneck(config, config.encoder_filters[-1])
        self.decoders = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)

    def forward(self, x):
        encoder_output, skip_connections = forward_encoders(self.config, self.encoders, x, domain_change=self.config.domain_change)
        bottleneck_output = forward_bottleneck(self.bottleneck, encoder_output, self.config)
        decoder_output = forward_decoders(self.config, self.decoders, bottleneck_output, skip_connections)
        return {'soi':decoder_output['soi'], 'logits':decoder_output['soi_logits']}

class WNet(nn.Module):
    def __init__(self, config, config_path=None, skip_type='concat'):
        super(WNet, self).__init__()
        self.config=config
        self.apply(lambda m: initialize_weights(m, config.weight_init_type, config.Is_weight_init))
        self.encoders = create_encoders(config)
        self.bottleneck = create_bottleneck(config, config.encoder_filters[-1])
        self.decoders1 = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)
        self.decoders2 = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)

    def forward(self, x):
        encoder_output, skip_connections = forward_encoders(self.config, self.encoders, x, domain_change=self.config.domain_change)
        bottleneck_output = forward_bottleneck(self.bottleneck, encoder_output, self.config)
        decoder_output1 = forward_decoders(self.config, self.decoders1, bottleneck_output, skip_connections)
        decoder_output2 = forward_decoders(self.config, self.decoders2, bottleneck_output, skip_connections)
        return {'soi':decoder_output1['soi'], 'interference':decoder_output2['soi'], 'logits':decoder_output1['soi_logits']}

class WNet_diff(nn.Module):
    def __init__(self, config, config_path=None, skip_type='concat'):
        super(WNet_diff, self).__init__()
        self.config=config
        self.apply(lambda m: initialize_weights(m, config.weight_init_type, config.Is_weight_init))
        self.encoders = create_encoders(config)
        self.bottleneck = create_bottleneck(config, config.encoder_filters[-1])
        self.decoders1 = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)
        self.decoders2 = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)

    def forward(self, x):
        encoder_output, skip_connections = forward_encoders(self.config, self.encoders, x, domain_change=self.config.domain_change)
        bottleneck_output = forward_bottleneck(self.bottleneck, encoder_output, self.config)
        decoder_output1 = forward_decoders(self.config, self.decoders1, bottleneck_output, skip_connections)
        decoder_output2 = forward_decoders(self.config, self.decoders2, encoder_output-bottleneck_output, skip_connections)
        return {'soi':decoder_output1['soi'], 'interference':decoder_output2['soi'], 'logits':decoder_output1['soi_logits']}


        
class VersaNet_Res(nn.Module):
    def __init__(self, config, config_path=None, skip_type='concat'):
        super(VersaNet_Res, self).__init__()
        self.config=config
        self.apply(lambda m: initialize_weights(m, config.weight_init_type, config.Is_weight_init))
        self.encoders = create_encoders(config)
        self.bottleneck = create_bottleneck(config, config.encoder_filters[-1])
        self.decoders = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)

    def forward(self, x):
        encoder_output, skip_connections = forward_encoders(self.config, self.encoders, x, domain_change=self.config.domain_change)
        bottleneck_output = forward_bottleneck(self.bottleneck, encoder_output, self.config)
        decoder_output = forward_decoders(self.config, self.decoders, encoder_output+bottleneck_output, skip_connections)
        return {'soi':decoder_output['soi'], 'logits':decoder_output['soi_logits']}


        
class VersaNet_Mask(nn.Module):
    def __init__(self, config, config_path=None, skip_type='concat'):
        super(VersaNet_Mask, self).__init__()
        self.config=config
        self.apply(lambda m: initialize_weights(m, config.weight_init_type, config.Is_weight_init))
        self.encoders = create_encoders(config)
        self.bottleneck = create_bottleneck(config, config.encoder_filters[-1])
        self.decoders = create_decoders(config, config.encoder_filters[: len(config.encoder_strides)], config.encoder_strides, config.ker_size, config.skip_type, config.channel_wise)

    def forward(self, x):
        encoder_output, skip_connections = forward_encoders(self.config, self.encoders, x, domain_change=self.config.domain_change)
        bottleneck_output = forward_bottleneck(self.bottleneck, encoder_output, self.config)
        decoder_output = forward_decoders(self.config, self.decoders, encoder_output*bottleneck_output, skip_connections)
        return {'soi':decoder_output['soi'], 'logits':decoder_output['soi_logits']}




class StraightThroughSignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def straight_through_sign(x):
    return StraightThroughSignFunction.apply(x) 


class QPSKDemod(nn.Module):
    def __init__(self, samples_per_symbol=16, num_symbols=None):
        super(QPSKDemod, self).__init__()
        
        self.matched_filter_layer = RRC_MatchedFilterLayer()
        self.downsample_layer = DownsampleLayer(samples_per_symbol=samples_per_symbol)
        self.bit_estimator_layer = BitEstimatorLayer()

    def forward(self, x):
        x = self.matched_filter_layer(x)
        x = self.downsample_layer(x[:,0,:]+(1j*x[:,1,:]))
        x = self.bit_estimator_layer(x)
        return x

        

class DownsampleLayer(nn.Module):
    def __init__(self, samples_per_symbol, offset=8, num_symbols=None, axis=-1):
        super(DownsampleLayer, self).__init__()
        self.samples_per_symbol = samples_per_symbol
        self.offset = offset
        self.num_symbols = num_symbols
        self.axis = axis

    def forward(self, x):
        # Calculate the indices to keep
        if self.num_symbols is not None:
            end_idx = self.offset + self.num_symbols * self.samples_per_symbol
            indices = torch.arange(self.offset, end_idx, self.samples_per_symbol, device=x.device)
        else:
            indices = torch.arange(self.offset, x.shape[self.axis], self.samples_per_symbol, device=x.device)
        
        # Perform the downsampling
        y = torch.index_select(x, dim=self.axis, index=indices)
        return y

class DifferentiableDownsampleLayer(nn.Module):
    def __init__(self, samples_per_symbol=16, axis=-1):
        super(DifferentiableDownsampleLayer, self).__init__()
        self.samples_per_symbol = samples_per_symbol
        self.offset = 8
        self.axis = axis
        channels = 2

class ThresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient back
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None

class GatingModule(nn.Module):
    def __init__(self, channels, threshold=0):
        super(GatingModule, self).__init__()
        self.pointwise_conv = nn.Conv1d(channels, channels, 1, groups=channels, bias=False)
        self.threshold = threshold

        # Initialize pointwise_conv weights and biases
        nn.init.constant_(self.pointwise_conv.weight, 1)

    def forward(self, x):
        gating_values = self.pointwise_conv(x)
        return ThresholdFunction.apply(gating_values, self.threshold) * x


        
def single_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout
    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )
    


def double_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

def GATED_double_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    layers = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        GatingModule(out_channels),  # Gating after the first conv layer
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        GatingModule(out_channels),  # Gating after the second conv layer
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    ]

    return nn.Sequential(*layers)
    
def conv_lstm(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    layers = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    ]
    
    # AdvancedTransformerBottleneck as the second layer
    layers.append(LSTMBottleneck(input_channels=out_channels, hidden_size=config.hidden_lstm_size, num_layers=config.num_lstm_layers, conv_dim=conv_dim))
    
    return nn.Sequential(*layers)


def conv_transformer(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    layers = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    ]
    
    # AdvancedTransformerBottleneck as the second layer
    layers.append(AdvancedTransformerBottleneck(input_channels=out_channels, hidden_size=2048, num_layers=2, conv_dim=conv_dim, nhead=8))
    
    return nn.Sequential(*layers)



def conv_tf_encoder(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple for 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    # First layer: Convolution
    layers = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    ]

    # Calculate d_model
    encoder_filters = [2,64,128,256]
    stage_n = encoder_filters.index(in_channels)
    d_model = 4096 // (2**stage_n)

    # Second layer: MyTransformerEncoderLayer
    layers.append(MyTransformerEncoderLayer(d_model=d_model, nhead=config.nhead, batch_first=True))

    return nn.Sequential(*layers)
    
def gated_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    class GatedBlock(nn.Module):
        def __init__(self):
            super(GatedBlock, self).__init__()
            self.conv1 = Conv(out_channels // 2, out_channels // 2, kernel_size, padding=padding_size, groups=groups_setting)
            self.norm1 = get_normalization_layer(config.normalization_type, out_channels // 2)
            self.conv2 = Conv(out_channels // 2, out_channels // 2, kernel_size, padding=padding_size, groups=groups_setting)
            self.norm2 = get_normalization_layer(config.normalization_type, out_channels // 2)
            self.gate = nn.Sigmoid()
            self.concat_layer = nn.Sequential()

        def forward(self, x):
            # Splitting the tensor into two halves
            x1, x2 = x.chunk(2, dim=1)
            # First half through convolution, normalization, and activation
            x1 = nn.LeakyReLU()(self.norm1(self.conv1(x1)))
            # Second half through convolution and gating
            x2 = self.gate(self.norm2(self.conv2(x2)))
            # Applying the gate
            return self.concat_layer(torch.cat((x1 * x2, x1 * (1 - x2)), dim=1))

    # First strided convolution
    layers = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        GatedBlock()
    ]

    return nn.Sequential(*layers)


def triple_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False, conv_dim=1):
    # Convert kernel_size and stride to tuple if they are not already, for the 2D case
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    groups_setting = out_channels if channel_wise else 1
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout
    
    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout2d(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )
    
def dilated_convs(config, in_channels, out_channels, kernel_size, padding_size, dilation_depth=10, stride=1):
    layers = []
    
    # Initial convolution without dilation
    layers.extend([
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        nn.Dropout2d(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    ])
    
    # Add dilated convolutions
    for i in range(1, dilation_depth+1):
        dilation = 2**i
        layers.extend([
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=dilation*padding_size, dilation=dilation),
            get_normalization_layer(config.normalization_type, out_channels),
            nn.LeakyReLU()
        ])
    
    return nn.Sequential(*layers)
class ResidualConvUnit(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, padding_size, stride=1, channel_wise=False):
        super(ResidualConvUnit, self).__init__()
        self.stride = stride
        self.change_in_channels = in_channels != out_channels
        groups_setting = out_channels if channel_wise else 1

        self.conv_relu_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, groups=groups_setting),
            get_normalization_layer(config.normalization_type, out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
            get_normalization_layer(config.normalization_type, out_channels),
            nn.LeakyReLU()
        )

        if self.change_in_channels or stride != 1:
            self.match_identity = nn.Conv1d(
                in_channels, out_channels, 1, stride=stride, padding=0
            )
            self.match_norm = get_normalization_layer(config.normalization_type, out_channels)
        else:
            self.match_identity = nn.Identity()
            self.match_norm = nn.Identity()

    def forward(self, x):
        identity = self.match_identity(x)
        identity = self.match_norm(identity)
        out = self.conv_relu_block(x)
        out += identity
        return out


def residual_block(config, in_channels, out_channels, kernel_size, stride, padding_size):
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding_size),
        get_normalization_layer(config.normalization_type, out_channels)
    ]
    shortcut = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, stride=stride),
        get_normalization_layer(config.normalization_type, out_channels)
    )
    return nn.Sequential(*layers), shortcut

def ffc_res(config, in_channels, out_channels, kernel_size, padding_size, stride=1, conv_dim=1):
    # Compute the number of middle channels as the average of in and out channels
    mid_channels = (in_channels + out_channels) // 2

    return nn.Sequential(
        FFCResnetBlock(in_c=in_channels, mid_c=mid_channels, out_c=out_channels, stride=stride, kernel_size=kernel_size, conv_dim=conv_dim),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

def ffc_res_v2(config, in_channels, out_channels, kernel_size, padding_size, ratio_gin=.5, ratio_gout=.5, stride=1, channel_wise=False,  conv_dim=1):
    # Convert kernel_size and stride to tuple for 2D
    if conv_dim == 2 and isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if conv_dim == 2 and isinstance(stride, int):
        stride = (stride, stride)

    # Channel-wise or regular grouping
    groups_setting = out_channels if channel_wise else 1

    # Choose appropriate Conv and Dropout layers based on conv_dim
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    # First part of double_conv
    first_conv = nn.Sequential(
        Conv(in_channels, out_channels, kernel_size, stride=1, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    )

    # ffc_res block
    mid_channels = (out_channels + out_channels) // 2
    ffc_res_block = nn.Sequential(
        FFCResnetBlock(in_c=out_channels, mid_c=mid_channels, out_c=out_channels, stride=1, kernel_size=kernel_size, ratio_gin=ratio_gin, ratio_gout=ratio_gout, conv_dim=conv_dim),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

    # Second part of double_conv
    second_conv = nn.Sequential(
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, stride=stride, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

    # Combine all parts
    return nn.Sequential(first_conv, ffc_res_block, second_conv)

# from transformer import MyTransformerEncoderLayer
class EncoderBlock(nn.Module):
    def __init__(self, config,  in_channels, out_channels, kernel_size, stride, padding_size, channel_wise=False):
        super(EncoderBlock, self).__init__()
        conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1
        self.config=config
        if config.enc_block_type == 'single_conv':
            self.conv = single_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'double_conv':
            self.conv = double_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'GATED_double_conv':
            self.conv = GATED_double_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'gated_conv':
            self.conv = gated_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'conv_transformer':
            self.conv = conv_transformer(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'conv_tf_encoder':
            self.conv = conv_tf_encoder(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'conv_lstm':
            self.conv = conv_lstm(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'triple_conv':
            self.conv = triple_conv(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'ResidualConvUnit':
            self.conv = ResidualConvUnit(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, channel_wise=channel_wise)
        elif config.enc_block_type == 'ffc_res0' or config.enc_block_type == 'ffc_res1':
            self.conv = ffc_res(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, conv_dim=conv_dim)
        elif config.enc_block_type == 'ffc_res2':
            self.in_channels = in_channels
            if self.in_channels==2:
                self.conv = ffc_res(config, 2*in_channels, out_channels, kernel_size, padding_size, stride=stride, conv_dim=conv_dim)
            else:
                self.conv = ffc_res(config, in_channels, out_channels, kernel_size, padding_size, stride=stride, conv_dim=conv_dim)
        elif config.enc_block_type == 'ffc_res3':
            self.in_channels = in_channels
            if self.in_channels==2:
                self.conv = ffc_res_v2(config, 2*in_channels, out_channels, kernel_size, padding_size,  ratio_gin=0.5, ratio_gout=0.0625, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
            else:
                self.conv = ffc_res_v2(config, in_channels, out_channels, kernel_size, padding_size,  ratio_gin=0.0625, ratio_gout=0.0625, stride=stride, channel_wise=channel_wise, conv_dim=conv_dim)
        elif config.enc_block_type == 'dilated_convs':
            self.conv = dilated_convs(config, in_channels, out_channels, kernel_size, padding_size, stride=stride)  # you might adjust dilation_depth based on your requirements
        else:
            raise ValueError(f"Unsupported enc_block_type: {enc_block_type}")
    def forward(self, x):
        if (self.config.enc_block_type == 'ffc_res2' or self.config.enc_block_type == 'ffc_res3') and self.in_channels==2:
            x = self.conv(torch.cat((x,x), dim=1))
            return x, x
        else:
            x = self.conv(x)
            return x, x   # one for forward and one for skip connection



def create_encoders(config):
    encoders = nn.ModuleList()
    prev_filters = config.input_channels if config.channel_operation == "no" else 1

    for idx, (filters, strides) in enumerate(zip(config.encoder_filters, config.encoder_strides)):
        ker_size = config.ker0_size if idx == 0 and hasattr(config, 'ker0_size') else config.ker_size
        padding_size = (ker_size - 1) // 2

        encoder = EncoderBlock(config, prev_filters, filters, ker_size, strides, padding_size, config.channel_wise)
        encoders.append(encoder)
        prev_filters = filters

    return encoders

    
def forward_encoders(config, encoders, x, domain_change="no", skip_domain_change="no"):
    x = pre_transformations(x, domain_change=config.domain_change, channel_operation=config.channel_operation)
    skip_connections = []
    for encoder in encoders:
        x, skip = encoder(x)
        skip_connections.append(skip)
    skip_connections = pre_transformations(skip_connections, domain_change=skip_domain_change)
    x = post_transformations(x, domain_change=config.domain_change, channel_operation=config.channel_operation)
    return x, skip_connections
    
    




class LSTMBottleneck(nn.Module):
    def __init__(self, input_channels=512, hidden_size=64, num_layers=2, conv_dim=1):
        super(LSTMBottleneck, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_dim = conv_dim
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, input_channels)  # 2 for bidirectional
        self.Identity_out=nn.Identity()

    def forward(self, x):
        if self.conv_dim == 2:
            batch_size, channels, height, width = x.size()
            # Flatten spatial dimensions and treat as sequential data
            x = x.view(batch_size, channels, -1)  # Shape is (batch, channels, height * width)
            x = x.permute(0, 2, 1)  # Now shape is (batch, height * width, channels)
        else:
            x = x.transpose(1, 2)  # For 1D, shape is (batch, length, channels)

        
        x, _ = self.lstm(x)  # Output shape: (batch, length, hidden_size * 2)
        x = self.linear(x)

        if self.conv_dim == 2:
            x = x.permute(0, 2, 1)  # Shape is (batch, channels, height * width)
            x = x.view(batch_size, channels, height, width)  # Reshape back to original spatial dimensions
        else:
            x = x.transpose(1, 2)  # For 1D, shape is (batch, channels, length)
        x=self.Identity_out(x)
        return x

class ViTBottleneck(nn.Module):
    def __init__(self, input_channels=512, hidden_size=64, num_layers=2, conv_dim=1):
        super(ViTBottleneck, self).__init__()
        self.vit_encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            out_chans=input_channels
        )

    def forward(self, x):
        # Directly pass the input through the ViT encoder
        x = self.vit_encoder(x)
        return x


class TransformerBottleneck(nn.Module):
    def __init__(self, input_channels=512, hidden_size=512, num_layers=8, conv_dim=1, nhead=16):
        super(TransformerBottleneck, self).__init__()
        self.conv_dim = conv_dim

        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_channels, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Linear layer to adjust the output back to the original channel size
        self.linear = nn.Linear(input_channels, input_channels)

    def forward(self, x):
        if self.conv_dim == 2:
            # Handling 2D data: Flatten spatial dimensions and treat as sequential data
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1)
            x = x.permute(0, 2, 1)
        else:
            # For 1D data, swap the channels and length dimensions
            x = x.transpose(1, 2)

        # Transformer encoder expects shape: (seq_length, batch, channels)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Swap back to (batch, seq_length, channels)

        x = self.linear(x)

        if self.conv_dim == 2:
            # Reshape back to original spatial dimensions for 2D data
            x = x.permute(0, 2, 1)
            x = x.view(batch_size, channels, height, width)
        else:
            # Swap back channels and length for 1D data
            x = x.transpose(1, 2)

        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class AdvancedTransformerBottleneck(nn.Module):
    def __init__(self, input_channels=512, hidden_size=2048, num_layers=2, conv_dim=1, nhead=8):
        super(AdvancedTransformerBottleneck, self).__init__()
        self.conv_dim = conv_dim

        self.pos_encoder = PositionalEncoding(d_model=input_channels)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_channels, 
            nhead=nhead, 
            dim_feedforward=hidden_size,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(input_channels, input_channels)
        self.norm = nn.LayerNorm(input_channels)

    def forward(self, x):
        if self.conv_dim == 2:
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1)
            x = x.permute(0, 2, 1)
        else:
            x = x.transpose(1, 2)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.permute(1, 0, 2)

        x = self.linear(x)

        if self.conv_dim == 2:
            x = x.permute(0, 2, 1)
            x = x.view(batch_size, channels, height, width)
        else:
            x = x.transpose(1, 2)

        return x        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HighlyAdvancedTransformerBottleneck(nn.Module):
    def __init__(self, input_channels=512, hidden_size=2048, num_layers=2, conv_dim=1, nhead=8):
        super(HighlyAdvancedTransformerBottleneck, self).__init__()
        self.conv_dim = conv_dim

        self.pos_encoder = PositionalEncoding(d_model=input_channels)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_channels, 
            nhead=nhead, 
            dim_feedforward=hidden_size,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.mlp = MLP(input_channels, hidden_size, input_channels)
        self.norm = nn.LayerNorm(input_channels)

    def forward(self, x):
        if self.conv_dim == 2:
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1)
            x = x.permute(0, 2, 1)
        else:
            x = x.transpose(1, 2)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        x = self.mlp(x)
        x = self.norm(x)

        if self.conv_dim == 2:
            x = x.permute(0, 2, 1)
            x = x.view(batch_size, channels, height, width)
        else:
            x = x.transpose(1, 2)

        return x
        
class SEBottleneck(nn.Module):
    def __init__(self, input_channels=512):
        super(SEBottleneck, self).__init__()
        self.se_block = monai.networks.blocks.SEBlock(
            spatial_dims=1,
            in_channels=input_channels,
            n_chns_1=input_channels,
            n_chns_2=input_channels,
            n_chns_3=input_channels,
            r=2,
            acti_type_1=('relu', {'inplace': True}),
            acti_type_2='sigmoid',
            acti_type_final=('relu', {'inplace': True})
        )

    def forward(self, x):
        return self.se_block(x)
        
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ASPPBottleneck(nn.Module):
    def __init__(self, input_channels=512):
        super(ASPPBottleneck, self).__init__()
        self.aspp_block = monai.networks.blocks.SimpleASPP(
            spatial_dims=1,
            in_channels=input_channels,
            conv_out_channels=input_channels // 4,
            kernel_sizes=(1, 3, 3, 3),
            dilations=(1, 2, 4, 6),
            norm_type='BATCH',
            acti_type='LEAKYRELU',
            bias=False
        )

    def forward(self, x):
        return self.aspp_block(x)

class WaveNetBottleneck(nn.Module):
    def __init__(self, wavenet_config, time_channels):
        super(WaveNetBottleneck, self).__init__()
        # Adjust the input and output channels of WaveNet
        wavenet_config.input_channels = time_channels

        self.wavenet = WaveNet_separator(wavenet_config)

    def forward(self, x):
        # WaveNet expects input of shape (batch, channel, length)
        wavenet_output = self.wavenet(x)['soi']
        return wavenet_output
        
def create_bottleneck(config, time_channels, freq_channels=None, kernel_size=3):
    conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1   
    layers = []
    if freq_channels:
        combined_channels = time_channels + freq_channels
        layers.append(nn.Conv1d(combined_channels, time_channels, config.kernel_size, padding=config.kernel_size//2))
        layers.append(nn.ReLU())
        layers.append(SEBlock(time_channels))  # Squeeze-and-Excitation block
        layers.append(nn.Conv1d(time_channels, time_channels, config.kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
    elif config.bottleneck_type == "lstm":  # Check if config chooses LSTM
        layers.append(LSTMBottleneck(time_channels, config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim))
    elif config.bottleneck_type == "TransformerBottleneck":  # Check if config chooses LSTM
        layers.append(TransformerBottleneck(time_channels, config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim))
    elif config.bottleneck_type == "AdvancedTransformerBottleneck":  # Check if config chooses LSTM
        layers.append(AdvancedTransformerBottleneck(time_channels, config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim))
    elif config.bottleneck_type == "HighlyAdvancedTransformerBottleneck":  # Check if config chooses LSTM
        layers.append(HighlyAdvancedTransformerBottleneck(time_channels, config.hidden_lstm_size, config.num_lstm_layers, conv_dim=conv_dim))
    elif config.bottleneck_type == "wave": 
        layers.append(WaveNetBottleneck(config, time_channels))
    elif config.bottleneck_type == "SEBottleneck":  
        layers.append(SEBottleneck(time_channels))
    elif config.bottleneck_type == "ASPP":  # Adding ASPP option
        layers.append(ASPPBottleneck(time_channels))    
    else:
        layers.append(nn.Identity())

    return nn.Sequential(*layers)

# def forward_bottleneck(bottleneck_layer, time_features, config, freq_features=None):
#     if freq_features is not None:  # If freq_features are provided
#         combined_features = torch.cat((time_features, freq_features), dim=1)
#         return bottleneck_layer(combined_features)
#     else:  # If freq_features are not provided
#         return bottleneck_layer(time_features)  # Directly pass through the identity layer.
def forward_bottleneck(bottleneck_layer, time_features, config, freq_features=None):
    bottleneck_input = time_features if freq_features is None else torch.cat((time_features, freq_features), dim=1)
    bottleneck_output = bottleneck_layer(bottleneck_input)

    # Check for 'bottleneck_structure' in config and apply the operation
    bottleneck_structure = getattr(config, 'bottleneck_structure', None)
    if bottleneck_structure == 'res':
        # Apply residual connection
        return bottleneck_input + bottleneck_output
    elif bottleneck_structure == 'mask':
        # Apply masking
        return bottleneck_input * bottleneck_output
    else:
        # No specific structure, pass the output as is
        return bottleneck_output




def single_convT(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in
    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    
    
    groups_setting = in_channels if channel_wise else 1
    return nn.Sequential(
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )


def double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in
    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout
    return nn.Sequential(
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(skip_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

def GATED_double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in

    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    layers = [
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        GatingModule(out_channels),  # Gating after the transposed conv layer
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(skip_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        GatingModule(out_channels),  # Gating after the standard conv layer
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    ]

    return nn.Sequential(*layers)
    
def convT_transformer(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in
    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    # First layer: ConvTranspose
    layers = [
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()
    ]

    # AdvancedTransformerBottleneck as the second layer
    layers.append(AdvancedTransformerBottleneck(input_channels=skip_channels, hidden_size=2048, num_layers=2, conv_dim=conv_dim, nhead=8))

    return nn.Sequential(*layers)
    
def triple_convT(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in
    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout
    return nn.Sequential(
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        Conv(skip_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Conv(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )

def ResidualConvUnitT(config, in_channels, out_channels, kernel_size, padding_size, stride, skip_channels, channel_wise=False):
    groups_setting = in_channels if channel_wise else 1
    change_in_channels = in_channels != out_channels
    layers = []

    conv_transpose = nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=stride-1, groups=groups_setting
    )
    normalization1 = get_normalization_layer(config.normalization_type, out_channels)
    activation1 = nn.LeakyReLU()
    dropout = nn.Dropout2d(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity()

    layers += [conv_transpose, normalization1, activation1, dropout]

    # Match dimensions of the skip connection and concatenate
    conv1 = nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting)
    normalization2 = get_normalization_layer(config.normalization_type, out_channels)
    activation2 = nn.LeakyReLU()

    # Add the second conv layer after concatenation
    layers += [conv1, normalization2, activation2]

    if change_in_channels or stride != 1:
        # Adjust the dimensions of identity to match the output
        match_identity = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=stride-1, groups=groups_setting
        )
        match_norm = get_normalization_layer(config.normalization_type, out_channels)
    else:
        match_identity = nn.Identity()
        match_norm = nn.Identity()

    # Add identity adjustment layers to the beginning of the sequential block
    layers = [match_identity, match_norm] + layers

    return nn.Sequential(*layers)

def ffc_resT(config, in_channels, out_channels, kernel_size, padding_size, stride, skip_channels, channel_wise=False):
    groups_setting = in_channels if channel_wise else 1
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        nn.Dropout2d(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        ffc_res(config, out_channels, out_channels, 3, 0, 1),
        nn.Conv1d(skip_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding_size, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU()
    )
    
def gated_convT(config, in_channels, out_channels, kernel_size, padding_size, stride_in, skip_channels, channel_wise=False, conv_dim=1):
    # Adjust kernel_size and stride for 2D if necessary
    if conv_dim == 2 and isinstance(kernel_size, int) and isinstance(stride_in, int):
        kernel_size, stride = (kernel_size, kernel_size), (stride_in, stride_in)
    else:
        kernel_size, stride = kernel_size, stride_in

    groups_setting = out_channels if channel_wise else 1
    ConvT = nn.ConvTranspose2d if conv_dim == 2 else nn.ConvTranspose1d
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d
    Dropout = nn.Dropout2d if conv_dim == 2 else nn.Dropout

    class GatedBlock(nn.Module):
        def __init__(self):
            super(GatedBlock, self).__init__()
            self.conv1 = Conv(skip_channels // 2, out_channels // 2, kernel_size, padding=padding_size, groups=groups_setting)
            self.norm1 = get_normalization_layer(config.normalization_type, out_channels // 2)
            self.conv2 = Conv(skip_channels // 2, out_channels // 2, kernel_size, padding=padding_size, groups=groups_setting)
            self.norm2 = get_normalization_layer(config.normalization_type, out_channels // 2)
            self.gate = nn.Sigmoid()

        def forward(self, x):
            x1, x2 = x.chunk(2, dim=1)
            x1 = nn.LeakyReLU()(self.norm1(self.conv1(x1)))
            x2 = self.gate(self.norm2(self.conv2(x2)))
            return torch.cat((x1 * x2, x1 * (1 - x2)), dim=1)

    # Define the layer sequence
    return nn.Sequential(
        ConvT(in_channels, out_channels, kernel_size, stride=stride, padding=padding_size, output_padding=-1+stride_in, groups=groups_setting),
        get_normalization_layer(config.normalization_type, out_channels),
        nn.LeakyReLU(),
        Dropout(p=config.spatial_dropout_rate) if config.spatial_dropout_rate > 0 else nn.Identity(),
        GatedBlock()
    )


class DecoderBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding_size, skip_type='concat', channel_wise=False):
        super(DecoderBlock, self).__init__()
        conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1
        setattr(config, 'dec_block_type', getattr(config, 'dec_block_type', getattr(config, 'enc_block_type')))
        self.skip_type = skip_type
        # Use the skip_channels method to determine the number of channels for the Conv1d layer in double_convT
        if config.dec_block_type == 'single_conv':
            self.conv_block = single_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'double_conv':
            self.conv_block = double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'GATED_double_convT':
            self.conv_block = GATED_double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'convT_transformer':
            self.conv_block = convT_transformer(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'gated_convT':
            self.conv_block = gated_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'triple_conv':
            self.conv_block = triple_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'ResidualConvUnit':
            self.conv_block = double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'ffc_res0' or config.dec_block_type == 'ffc_res2' or config.dec_block_type == 'ffc_res3':
            self.conv_block = double_convT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        elif config.dec_block_type == 'ffc_res1':
            self.conv_block = ffc_resT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False, conv_dim=conv_dim)
        else:
            raise ValueError(f"Unrecognized decoder block type: {config.dec_block_type}")
# (config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False)            # self.conv_block = ResidualConvUnitT(config, in_channels, out_channels, kernel_size, padding_size, stride, self.skip_channels(out_channels), channel_wise=False)
            
    
    def skip_handler(self, x, skip_connection):
        if self.skip_type == 'concat':
            return torch.cat([x, skip_connection], dim=1)
        elif self.skip_type == 'add':
            return x+ skip_connection
        else:
            raise ValueError(f"Unknown skip type: {self.skip_type}")
    
    def skip_channels(self, out_channels):
        # Determine the number of channels that the skip_handler will output
        if self.skip_type == 'concat':
            return 2 * out_channels
        elif self.skip_type == 'add':
            return out_channels
        else:
            return out_channels

    def forward(self, x, skip_connection):
        x = self.conv_block[0:3](x)
        x = self.skip_handler(x, skip_connection)
        x = self.conv_block[3:](x)
        return x




        
def create_decoders(config, encoder_filters, encoder_strides, ker_size, skip_type, channel_wise=False):
    conv_dim = config.conv_dim if hasattr(config, 'conv_dim') else 1
    padding_size = (ker_size - 1) // 2
    decoders = nn.ModuleList()
    Conv = nn.Conv2d if conv_dim == 2 else nn.Conv1d

    output_kernel_size = (1, 1) if conv_dim == 2 else 1 

    
    for encoder_stride, filters, ConvTranspose1d_filters in zip(reversed(encoder_strides[1:]), reversed(encoder_filters[:-1]), reversed(encoder_filters[1:])):
        decoder = DecoderBlock(config, ConvTranspose1d_filters, filters, ker_size, encoder_stride, padding_size, skip_type=skip_type, channel_wise=False)
        decoders.append(decoder)
    
    # Adding the decoder's output layer here
    decoder_output_layer = Conv(encoder_filters[0],   config.output_channels if config.channel_operation=="no" else 1 , kernel_size=output_kernel_size)
    decoders.append(decoder_output_layer)
    
    return decoders

def forward_decoders(config, decoders, x, skip_connections):
    *actual_decoders, output_layer = decoders
    x = pre_transformations(x, domain_change=config.domain_change, channel_operation=config.channel_operation)

    for decoder, skip in zip(actual_decoders, reversed(skip_connections[:-1])):
        x = decoder(x, skip)
    logits=x
    # Using the output layer here
    x = output_layer(logits)
    # Then proceed to the domain transform
    decoder_output = post_transformations(x, domain_change=config.domain_change, channel_operation=config.channel_operation)
    return {'soi': decoder_output, 'soi_logits': logits}




def create_output():
    return nn.Identity()

def forward_output(output_layer, x):
    return output_layer(x)




def stft_transform(x, n_fft, hop_length, mode='direct'):
    if mode == 'direct':
        x = x[:, 0, :] + (1j * x[:, 1, :])
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True, center=False)
        x = x.unsqueeze(1)
        x = torch.cat((x.real, x.imag), 1)
    elif mode == 'inverse':
        x = x[:, 0, :, :] + (1j * x[:, 1, :, :])
        x = torch.istft(x, n_fft=n_fft, hop_length=hop_length, return_complex=True, center=False)
        x = x.unsqueeze(1)
        x = torch.cat((x.real, x.imag), 1)
    return x
def polar_transform(x, mode='direct'):
    if mode == 'direct':
        # Cartesian to Polar
        real = x[:, 0, :]
        imag = x[:, 1, :]
        magnitude = torch.sqrt(real**2 + imag**2).unsqueeze(1)
        phase = torch.atan2(imag, real).unsqueeze(1)
        x = torch.cat((magnitude, phase), 1)
    elif mode == 'inverse':
        # Polar to Cartesian
        magnitude = x[:, 0, :]
        phase = x[:, 1, :]
        real = (magnitude * torch.cos(phase)).unsqueeze(1)
        imag = (magnitude * torch.sin(phase)).unsqueeze(1)
        x = torch.cat((real, imag), 1)
    return x
def fft_transform(x, mode='direct'):
    if mode == 'direct':
        # Combine real and imaginary parts to create complex-valued input
        x_complex = x[:, 0, :] + (1j * x[:, 1, :])
        # Perform FFT
        x_fft = torch.fft.fft(x_complex)
        # Split into real and imaginary parts
        x_real = x_fft.real.unsqueeze(1)
        x_imag = x_fft.imag.unsqueeze(1)
        x = torch.cat((x_real, x_imag), 1)
    elif mode == 'inverse':
        # Combine real and imaginary parts to create complex-valued input
        x_complex = x[:, 0, :] + (1j * x[:, 1, :])
        # Perform inverse FFT
        x_ifft = torch.fft.ifft(x_complex)
        # Split into real and imaginary parts
        x_real = x_ifft.real.unsqueeze(1)
        x_imag = x_ifft.imag.unsqueeze(1)
        x = torch.cat((x_real, x_imag), 1)
    return x

def identity_transform(x, mode='direct'):
    return x

def get_transform_function(config, mode='direct'):
    transform_type = config.get('transform_type', 'identity')
    if transform_type == 'stft':
        return lambda x: stft_transform(x, n_fft=config.n_fft, hop_length=config.hop_length, mode=mode)
    if transform_type == 'hankel':
        return lambda x: hankel_transform(x, L_win=config.n_fft, hop_length=config.hop_length, mode=mode)
    elif transform_type == 'fft':
        return lambda x: fft_transform(x, mode=mode)
    elif transform_type == 'polar':
        return lambda x: polar_transform(x, mode=mode)
    else:  # Default to identity transform
        return identity_transform





def initialize_weights(model, init_type=None,Is_weight_init=False):
    if Is_weight_init:
        for m in model.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if init_type == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif init_type == 'he':
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight)
                elif init_type == 'sparse':
                    init.sparse_(m.weight, sparsity=0.9)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def compute_avg_l0_norm(parameters):
    total_non_zero = 0
    total_elements = 0

    for param in parameters:
        # Count non-zero elements
        total_non_zero += param.ne(0).int().sum()

        # Count total elements
        total_elements += param.numel()

    # Calculate average L0 norm
    avg_l0_norm = total_non_zero.float() / total_elements if total_elements > 0 else 0
    return avg_l0_norm.item()  # Convert to Python float



def get_normalization_layer(normalization_type, channels):
    if normalization_type == "batch":
        return nn.BatchNorm1d(channels)
    elif normalization_type == "layer":
        return nn.LayerNorm(channels)
    elif normalization_type == "group":
        return nn.GroupNorm(num_groups=4, num_channels=channels)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization_type}")


def pre_transformations(input_tensor, domain_change="no", channel_operation="no"):
    if domain_change == "fft":
        # Ensure the input tensor has 2 channels for real and imaginary parts
        if input_tensor.size(1) != 2:
            raise ValueError(f"Expected 2 channels for real and imaginary parts, but got {input_tensor.size(1)} channels.")
        # Forward FFT: channel to complex -> fft -> complex to channel
        complex_tensor = channels_to_complex(input_tensor)
        fft_result = torch.fft.fft(complex_tensor, dim=-1, norm=None)
        input_tensor = complex_to_channels(fft_result)

    if channel_operation == "merge":
        # Merge channels: (batch, channel=2, length) -> (batch x channel, 1, length)
        batch, channels, length = input_tensor.shape
        input_tensor = input_tensor.view(batch * channels, 1, length)
    
    return input_tensor


def post_transformations(input_tensor, domain_change="no", channel_operation="no"):
    if channel_operation == "merge":
        # Split the merged channels: (batch*channel, 1, length) -> (batch, channel, length)
        # Assuming channel is 2 after a previous merge operation
        merged_dim = input_tensor.size(0)
        if merged_dim % 2 != 0:
            raise ValueError(f"The merged dimension {merged_dim} is not divisible by the number of channels (2).")
        batch = merged_dim // 2
        input_tensor = input_tensor.view(batch, 2, -1)

    if domain_change == "fft":
        # Inverse FFT: channel to complex -> ifft -> complex to channel
        if input_tensor.size(1) != 2:
            raise ValueError(f"Expected 2 channels for real and imaginary parts, but got {input_tensor.size(1)} channels.")
        complex_tensor = channels_to_complex(input_tensor)
        ifft_result = torch.fft.ifft(complex_tensor, dim=-1, norm=None)
        input_tensor = complex_to_channels(ifft_result)
    
    return input_tensor
