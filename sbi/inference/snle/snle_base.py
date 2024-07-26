# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.base import NeuralInference
from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.neural_nets import ConditionalDensityEstimator, likelihood_nn
from sbi.neural_nets.density_estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.utils import check_estimator_arg, check_prior, x_shape_from_simulation

from alive_progress import alive_bar
import numpy as np

class LikelihoodEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for Sequential Neural Likelihood Estimation methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = likelihood_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: str = "SNLE",
        data_device: Optional[str] = None,
    ) -> "LikelihoodEstimator":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. If `False`, SNLE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        # pyright false positive, will be fixed with pyright 1.1.310
        return super().append_simulations(  # type: ignore
            theta=theta,
            x=x,
            exclude_invalid_x=exclude_invalid_x,
            from_round=from_round,
            algorithm=algorithm,
            data_device=data_device,
        )

    def train(
        self,
        train_dataloader,
        test_dataloader,
        optimizer,
        optimizer_parameter,
        summary_net = None,
        loss_summary_net = None,
        train_summary_net_freezed_rounds = 0,
        #pretrain_summary_net = False,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        retrain_from_scratch: bool = False,
    ) -> ConditionalDensityEstimator:
        r"""Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that has learned the distribution $p(x|\theta)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        train_loader, val_loader = train_dataloader, test_dataloader
        
        if summary_net is not None:
            self.sum_net = True
        else:
            self.sum_net = False    
            
        self.epoch = 0

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _ = train_loader.dataset[0]
            theta, x = theta.unsqueeze(0), x.unsqueeze(0)
            # Use only training data for building the neural net (z-scoring transforms)
            if self.sum_net:
                x = summary_net.to('cpu')(x)
                
                
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )
            assert (
                len(x_shape_from_simulation(x.to("cpu"))) < 3
            ), "SNLE cannot handle multi-dimensional simulator output."
            del theta, x

        self._neural_net.to(self._device)
        summary_net.to(self._device)    
        
        if self.sum_net:
            #self.optimizer_le = optimizer(self._neural_net.parameters(), **optimizer_parameter)
            #self.optimizer_sn = optimizer(summary_net.parameters(), **optimizer_parameter)
            self.optimizer = optimizer( list(summary_net.parameters()) + list(self._neural_net.parameters()), **optimizer_parameter)
        else:
            self.optimizer = optimizer(self._neural_net.parameters(), **optimizer_parameter)
        
        if loss_summary_net is None:
            loss_summary_net = torch.nn.MSELoss()
        
        train_loss_summary_net = []
        train_loss_density_net = []
        test_loss_summary_net = []
        test_loss_density_net = []
        
        with alive_bar(max_num_epochs, force_tty=True) as bar:
            while self.epoch <= max_num_epochs and not self._converged(
                self.epoch, stop_after_epochs
            ):
                # Train for a single epoch.
                self._neural_net.train()
                if self.sum_net:
                    summary_net.train()
                    
                temp_loss_sum = []
                temp_loss_de = []
                
                for batch in train_loader:
                    self.optimizer.zero_grad()
                    theta_batch, x_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                    )
                    
                    # trainingsloop with summary-net freezed
                    if self.epoch < train_summary_net_freezed_rounds and self.sum_net:
                        x_batch = summary_net(x_batch).detach()
                    else:
                        x_batch = summary_net(x_batch)
                        
                    # Calculate loss.
                    temp_loss_sum.append(loss_summary_net(x_batch, theta_batch).mean().item())
                    
                    # Evaluate on x with theta as context.
                    if self.epoch >= train_summary_net_freezed_rounds and self.sum_net:
                        '''
                        Now: 
                        pretrain: self._loss(theta=theta_batch, x=x_batch)
                        then: self._loss(theta=x_batch, x=theta_batch)
                        does not work well, convergence but biased with large variance
                        
                        Next: 
                        pretrain:
                        self._loss(theta=theta_batch, x=x_batch)
                        then: (self._loss(theta=theta_batch, x=x_batch) + self._loss(theta=x_batch, x=theta_batch))*0.5
                        sharp distribution, slightly biased, but works well
                        
                        Next:
                        pretrain: train_loss = self._loss(theta=theta_batch, x=x_batch)
                        then: de: train_loss = self._loss(theta=theta_batch, x=x_batch) sn: train_loss = self._loss(theta=x_batch, x=theta_batch)
                        
                        Next:
                        pretrain: train_loss = self._loss(theta=theta_batch, x=x_batch)
                        then: de: train_loss = self._loss(theta=theta_batch, x=x_batch) sn: summary_net_loss = loss_summary_net(x_batch, samples)
                        '''
                        train_loss = (self._loss(theta=theta_batch, x=x_batch) + self._loss(theta=x_batch, x=theta_batch))*0.5
                    else:
                        train_loss = self._loss(theta=theta_batch, x=x_batch)
                    
                    train_loss = torch.mean(train_loss)
                    temp_loss_de.append(train_loss.item())

                    train_loss.backward()
                    if clip_max_norm is not None:
                        clip_grad_norm_(
                            self._neural_net.parameters(),
                            max_norm=clip_max_norm,
                        )
                        if self.sum_net:
                            clip_grad_norm_(
                                summary_net.parameters(),
                                max_norm=clip_max_norm,
                            )
                    self.optimizer.step()


                train_loss_summary_net.append(np.mean(temp_loss_sum))
                train_loss_density_net.append(np.mean(temp_loss_de))    

                # Calculate validation performance.
                self._neural_net.eval()
                summary_net.eval()
                with torch.no_grad():
                    
                    temp_loss_sum = []
                    temp_loss_de = []
                    test_blub = []
                    
                    
                    for batch in val_loader:
                        theta_batch, x_batch = (
                            batch[0].to(self._device),
                            batch[1].to(self._device),
                        )
                        
                        # trainingsloop with summary-net freezed
                        if self.sum_net:
                            x_batch = summary_net(x_batch)
                            
                        # Calculate loss.
                        temp_loss_sum.append(loss_summary_net(x_batch, theta_batch).mean().item())
                        
                        # Evaluate on x with theta as context.
                        if self.epoch >= train_summary_net_freezed_rounds and self.sum_net:    
                            val_losses = train_loss = (self._loss(theta=theta_batch, x=x_batch) + self._loss(theta=x_batch, x=theta_batch))*0.5
                        else:
                            val_losses = self._loss(theta=theta_batch, x=x_batch)
                            test_blub.append(0)
                        temp_loss_de.append(val_losses.mean().item())

                test_loss_summary_net.append(np.mean(temp_loss_sum))
                test_loss_density_net.append(np.mean(temp_loss_de))  
                print(f"{test_loss_density_net[-1]=}", f"{test_loss_summary_net[-1]=}", f'{np.mean(test_blub)=}')
                self.epoch += 1
                bar()

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return self._neural_net, summary_net, (train_loss_summary_net, train_loss_density_net, test_loss_summary_net, test_loss_density_net)

    def build_posterior(
        self,
        density_estimator: Optional[ConditionalDensityEstimator] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        importance_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[
        MCMCPosterior, RejectionPosterior, VIPosterior, ImportanceSamplingPosterior
    ]:
        r"""Build posterior from the neural density estimator.

        SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
        p(\theta)$ and draw samples from the posterior with MCMC or rejection sampling.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
            initialization `inference = SNLE(prior)` or to `.build_posterior
            (prior=prior)`."""
            prior = self._prior
        else:
            check_prior(prior)

        if density_estimator is None:
            likelihood_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            likelihood_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = str(next(density_estimator.parameters()).device)

        potential_fn, theta_transform = likelihood_estimator_based_potential(
            likelihood_estimator=likelihood_estimator,
            prior=prior,
            x_o=None,
        )

        if sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                **mcmc_parameters or {},
            )
        elif sample_with == "rejection":
            self._posterior = RejectionPosterior(
                potential_fn=potential_fn,
                proposal=prior,
                device=device,
                **rejection_sampling_parameters or {},
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                **vi_parameters or {},
            )
        elif sample_with == "importance":
            self._posterior = ImportanceSamplingPosterior(
                potential_fn=potential_fn,
                proposal=prior,
                device=device,
                **importance_sampling_parameters or {},
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)

    def _loss(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Return loss for SNLE, which is the likelihood of $-\log q(x_i | \theta_i)$.

        Returns:
            Negative log prob.
        """
        theta = reshape_to_batch_event(
            theta, event_shape=self._neural_net.condition_shape
        )
        x = reshape_to_batch_event(x, event_shape=self._neural_net.input_shape)
        return self._neural_net.loss(x, condition=theta)
