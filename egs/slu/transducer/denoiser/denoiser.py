# Author : Sonal Joshi, based on intial scripts by Saurabh Kataria

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import yaml
from .conv_tasnet import TasNet
from sklearn.preprocessing import StandardScaler
from torch import nn

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)


class DenoiserReconstruction(nn.Module):
    def __init__(self,  denoiser, device):
        super().__init__()
        self.denoiser = denoiser
        self.device = device

    def forward(self, audio):
        """
        Passing wavefrom through Denoiser Defender.
        """

        # Added for processing single audio file as in deepspeech armory [Sonal 29Oct20]
        if audio.ndim == 1:
            num_samples = audio.shape[0]
            recording = audio.unsqueeze(dim=0).to(self.device)

            # Setup inputs
            reconstructed_audio = self.denoiser(recording)
            return reconstructed_audio.squeeze().to(self.device)

        else:
            reconstructions = []
            num_samples = audio.shape[1]
            for idx in range(audio.shape[0]):
                recording = audio.unsqueeze(dim=0).to(self.device)

                # Setup inputs
                reconstructed_audio = self.denoiser(recording)
                reconstructed_audio = reconstructed_audio.squeeze().to(self.device)
                reconstructions.append(reconstructed_audio)
                
            #return torch.stack(reconstructions) 
            return torch.stack(reconstructions)
    
    @staticmethod
    def backward(ctx, grad_output):
        """BPDA grad_x=1
        """
        # we need one output per input
        return grad_output.cpu(), None, None, None, None, None, None, None # Added .cpu() by Sonal to avoid CUDA error



class DenoiserDefender(nn.Module):
    def __init__(self, denoiser_model_dir : Path, denoiser_model_ckpt : Path,  device: str):
        super().__init__()
        self.device = device
        with open(denoiser_model_dir / 'config.yml') as f:
           self.config = yaml.load(f, Loader=yaml.Loader)

        logging.info(f"Denoiser parameter : num_spk = {self.config['num_spk']}")
        logging.info(f"Denoiser parameter : layer = {self.config['layer']}")
        logging.info(f"Denoiser parameter : stack = {self.config['stack']}")
        logging.info(f"Denoiser parameter : win = {self.config['win']}")
        logging.info(f"Denoiser parameter : TCN_dilationFactor = {self.config['TCN_dilationFactor']}")
        logging.info(f"Denoiser parameter : load_string = {self.config['load_model_string']}")

        self.model =  TasNet(num_spk=self.config['num_spk'], layer=self.config['layer'], enc_dim=self.config['enc_dim'], stack=self.config['stack'], kernel=self.config['kernel'], win=self.config['win'], TCN_dilationFactor=self.config['TCN_dilationFactor'])
        model_path = denoiser_model_dir / denoiser_model_ckpt

        load_string = self.config["load_model_string"]
        logging.info(f"Loading model : {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))[load_string]
        for key in list(state_dict):
            if key.startswith('module.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        for key in list(state_dict):
            if key.startswith('reconstructor.denoiser.'):
                state_dict.pop(key)
            if key.startswith('model.'):
                state_dict['.'.join(key.split('.')[1:])] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.reconstructor = DenoiserReconstruction(self.model,self.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.reconstructor(audio)