from math import exp, log, sqrt
from typing import Tuple, Union, Optional, Callable

import torch
from torch import Tensor, nn

from sbi.neural_nets.vf_estimators import VectorFieldEstimator
from sbi.types import Shape


class ScoreEstimator(VectorFieldEstimator):
    r"""Score estimator for score-based generative models (e.g., denoising diffusion).
    """
    def __init__(
            self, 
            net: nn.Module,
            condition_shape: torch.Size,            
            weight_fn: Union[str, Callable]
            ) -> None:
        """
        Class for score estimators with variance exploding (NCSN), variance preserving (DDPM), or sub-variance preserving SDEs.
        """        
        super().__init__(net, condition_shape)
        if net is None:
            # Define a simple torch MLP network if not provided.
            nn.MLP()

        elif isinstance(net, nn.Module):
            self.net = net

        self.condition_shape = condition_shape
        # Set mean and standard deviation functions based on the type of SDE and noise bounds.
        self._set_mean_std_fn(sde_type, noise_minmax)


        # Set lambdas (variance weights) function
        self._set_weight_fn(weight_fn)
        
    def mean_fn(self, x0, times):
        raise NotImplementedError
    
    def std_fn(self, times):
        raise NotImplementedError

        # Set drift and diffusion function
        self._set_drift_diffusion_fn()

        self.mean = (
            0.0  # this still needs to be computed (mean of the noise distribution)
        )
        self.std = 1.0  # same

    def forward(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        score = self.net(input, condition, times)
        # Divide by standard deviation to mirror target score
        std = self.std_fn(times)
        return score / std
    
    def loss(self, input: Tensor, condition: Tensor) -> Tensor:
        """Denoising score matching loss (Song et al., ICLR 2021)."""
        # Sample diffusion times.
        times = torch.rand((input.shape[0],))

        # Sample noise
        eps = torch.randn_like(input)
        
        # Compute mean and standard deviation.
        mean = self.mean_fn(input, times)
        std = self.std_fn(times)

        # Get noised input, i.e., p(xt|x0)
        input_noised = mean + std * eps

        # Compute true score: -(mean - noised_input) / (std**2)
        score_target = -eps / std

        # Predict score.
        score_pred = self.forward(input_noised, condition, times)

        # Compute weights over time.
        weights = self.weight_fn(std)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_target - score_pred).pow(2.0), axis=-1)
        loss = torch.mean(weights * loss)

        return loss    

    def _set_weight_fn(self, weight_fn):
        """Get the weight function."""
        if weight_fn == "identity":
            self.weight_fn = lambda sigma: 1
        elif weight_fn == "variance":
            # From Song & Ermon, NeurIPS 2019.
            self.weight_fn = lambda sigma: sigma.pow(2.0)
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")
    
class VPScoreEstimator(ScoreEstimator):
    """ Class for score estimators with variance preserving SDEs (i.e., DDPM)."""
    def __init__(
            self, 
            net: nn.Module,
            condition_shape: torch.Size,
            weight_fn: Union[str, Callable]='variance',
            beta_min: float=0.1,
            beta_max: float=20.,                        
            ) -> None:        
        self.beta_min = beta_min
        self.beta_max = beta_max        
        super().__init__(net, condition_shape, weight_fn=weight_fn)
        
    def mean_fn(self, x0, times):
        return torch.exp(-0.25*times.pow(2.)*(self.beta_max-self.beta_min)-0.5*times*self.beta_min)*x0
        
    def std_fn(self, times):            
        return 1.-torch.exp(-0.5*times.pow(2.)*(self.beta_max-self.beta_min)-times*self.beta_min)
    
    def _beta_schedule(self, times):
        return self.beta_min + (self.beta_max - self.beta_min) * times
    
class subVPScoreEstimator(ScoreEstimator):
    """ Class for score estimators with sub-variance preserving SDEs."""
    def __init__(
            self, 
            net: nn.Module,
            condition_shape: torch.Size,
            weight_fn: Union[str, Callable]='variance',
            beta_min: float=0.1,
            beta_max: float=20.,                        
            ) -> None:        
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)
        
    def mean_fn(self, x0, times):
        return torch.exp(-0.25*times.pow(2.)*(self.beta_max-self.beta_min)-0.5*times*self.beta_min)*x0
    
    def std_fn(self, times):
        return (1.-torch.exp(-0.5*times.pow(2.)*(self.beta_max-self.beta_min)-times*self.beta_min)).power(2.)
    
    def _beta_schedule(self, times):
        return self.beta_min + (self.beta_max - self.beta_min) * times


class VEScoreEstimator(ScoreEstimator):
    """ Class for score estimators with variance exploding SDEs (i.e., SMLD)."""
    def __init__(
            self,
            net: nn.Module,
            condition_shape: torch.Size,
            weight_fn: Union[str, Callable]='variance',
            sigma_min: float=0.01,
            sigma_max: float=10.,                        
            ) -> None:        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(net, condition_shape, weight_fn=weight_fn)

    def mean_fn(self, x0, times):
        return x0
    
    def std_fn(self, times):
        return self.sigma_min.pow(2.) * (self.sigma_max / self.sigma_min).pow(2.*times)    
    
    def _sigma_schedule(self, times):
        return self.sigma_min * (self.sigma_max / self.sigma_min).pow(times)