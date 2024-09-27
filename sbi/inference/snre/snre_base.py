# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor, eye, nn, ones, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from sbi.inference.base import NeuralInference
from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.potentials import ratio_estimator_based_potential
from sbi.neural_nets import classifier_nn
from sbi.utils import (
    check_estimator_arg,
    check_prior,
    clamp_and_warn,
    x_shape_from_simulation,
)
from sbi.utils.torchutils import repeat_rows


class RatioEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        classifier: Union[str, Callable] = "resnet",
        device: str = "cpu",
        logging_level: Union[int, str] = "warning",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Sequential Neural Ratio Estimation.

        We implement three inference methods in the respective subclasses.

        - SNRE_A / AALR is limited to `num_atoms=2`, but allows for density evaluation
          when training for one round.
        - SNRE_B / SRE can use more than two atoms, potentially boosting performance,
          but allows for posterior evaluation **only up to a normalizing constant**,
          even when training only one round.
        - BNRE is a variation of SNRE_A aiming to produce more conservative posterior
          approximations.
        - SNRE_C / NRE-C is a generalization of SNRE_A and SNRE_B which can use multiple
          classes (similar to atoms) but encourages an exact likelihood-to-evidence
          ratio (density evaluation) by introducing an independently drawn class.
          Addressing the issue in SNRE_B which only estimates the ratio up to a function
          (normalizing constant) of the data $x$.

        Args:
            classifier: Classifier trained to approximate likelihood ratios. If it is
                a string, use a pre-configured network of the provided type (one of
                linear, mlp, resnet). Alternatively, a function that builds a custom
                neural network can be provided. The function will be called with the
                first batch of simulations (theta, x), which can thus be used for shape
                inference and potentially for z-scoring. It needs to return a PyTorch
                `nn.Module` implementing the classifier.

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
        check_estimator_arg(classifier)
        if isinstance(classifier, str):
            self._build_neural_net = classifier_nn(model=classifier)
        else:
            self._build_neural_net = classifier

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        algorithm: str = "SNRE",
        data_device: Optional[str] = None,
    ) -> "RatioEstimator":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. If `False`, SNRE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNRE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

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
        num_atoms: int = 10,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
    ) -> nn.Module:
        r"""Return classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            exclude_invalid_x: Whether to exclude simulation outputs `x=NaN` or `x=±∞`
                during training. Expect errors, silent or explicit, when `False`.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).
            loss_kwargs: Additional or updated kwargs to be passed to the self._loss fn.

        Returns:
            Classifier that approximates the ratio $p(\theta,x)/p(\theta)p(x)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        train_loader, val_loader = train_dataloader, test_dataloader

        clipped_batch_size = min(train_loader.batch_size, val_loader.batch_size)  # type: ignore

        if summary_net is not None:
            self.sum_net = True
        else:
            self.sum_net = False   
            
        num_atoms = int(
            clamp_and_warn(
                "num_atoms", num_atoms, min_val=2, max_val=clipped_batch_size
            )
        )

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
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            del x, theta
        self._neural_net.to(self._device)
        summary_net.to(self._device)
        
        if loss_summary_net is None:
            loss_summary_net = torch.nn.MSELoss()

        if self.sum_net:
            self.optimizer = optimizer( list(summary_net.parameters()) + list(self._neural_net.parameters()), **optimizer_parameter)
        else:
            self.optimizer = optimizer(self._neural_net.parameters(), **optimizer_parameter)
        
        train_loss_summary_net = []
        train_loss_density_net = []
        test_loss_summary_net = []
        test_loss_density_net = []
        self.epoch = 0
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
                    if self.sum_net:
                        x_batch = summary_net(x_batch)

                    train_losses = self._loss(
                        theta_batch, x_batch, num_atoms, **loss_kwargs
                    )
                    train_loss = torch.mean(train_losses)

                    temp_loss_sum.append(loss_summary_net(x_batch, theta_batch).mean().item())
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

                self.epoch += 1

                train_loss_summary_net.append(np.mean(temp_loss_sum))
                train_loss_density_net.append(np.mean(temp_loss_de))    
                temp_loss_sum = []
                temp_loss_de = []

                # Calculate validation performance.
                self._neural_net.eval()
                if self.sum_net:
                    summary_net.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        theta_batch, x_batch = (
                            batch[0].to(self._device),
                            batch[1].to(self._device),
                        )
                        
                        if self.sum_net:
                            x_batch = summary_net(x_batch)
                            
                        temp_loss_sum.append(loss_summary_net(x_batch, theta_batch).mean().item())
                        
                        val_losses = self._loss(
                            theta_batch, x_batch, num_atoms, **loss_kwargs
                        )
                        temp_loss_de.append(torch.mean(val_losses).item())
                    # Take mean over all validation samples.
                test_loss_summary_net.append(np.mean(temp_loss_sum))
                test_loss_density_net.append(np.mean(temp_loss_de))  
                bar()


        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return self._neural_net, summary_net, (train_loss_summary_net, train_loss_density_net, test_loss_summary_net, test_loss_density_net)

    def _classifier_logits(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        """Return logits obtained through classifier forward pass.

        The logits are obtained from atomic sets of (theta,x) pairs.
        """
        batch_size = theta.shape[0]
        repeated_x = repeat_rows(x, num_atoms)

        # Choose `1` or `num_atoms - 1` thetas from the rest of the batch for each x.
        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        return self._neural_net(atomic_theta, repeated_x)

    @abstractmethod
    def _loss(self, theta: Tensor, x: Tensor, num_atoms: int) -> Tensor:
        raise NotImplementedError

    def build_posterior(
        self,
        density_estimator: Optional[nn.Module] = None,
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

        SNRE trains a neural network to approximate likelihood ratios. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
        p(\theta)$ and draw samples from the posterior with MCMC or rejection sampling.
        Note that, in the case of single-round SNRE_A / AALR, it is possible to
        evaluate the log-probability of the **normalized** posterior, but sampling
        still requires MCMC (or rejection sampling).

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
                that some of the methods admit a `mode seeking` property (e.g. rKL)
                whereas some admit a `mass covering` one (e.g fKL).
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
                initialization `inference = SNRE(prior)` or to `.build_posterior
                (prior=prior)`."""
            prior = self._prior
        else:
            check_prior(prior)

        if density_estimator is None:
            ratio_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            ratio_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = str(next(density_estimator.parameters()).device)

        potential_fn, theta_transform = ratio_estimator_based_potential(
            ratio_estimator=ratio_estimator,
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
