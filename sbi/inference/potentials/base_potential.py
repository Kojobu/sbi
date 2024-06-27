# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.utils.user_input_checks import process_x


class BasePotential(metaclass=ABCMeta):
    def __init__(
        self,
        prior: Optional[Distribution],
        x_o: Optional[Tensor] = None,
        device: str = "cpu",
    ):
        """Initialize potential function.

        This parent class takes care of setting `x_o`.

        Args:
            prior: Prior distribution.
            x_o: Observed data.
            device: Device on which to evaluate the potential function.
        """
        self.device = device
        self.prior = prior
        self.set_x(x_o)

    @abstractmethod
    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        raise NotImplementedError

    @property
    def x_is_iid(self) -> bool:
        """If x has batch dimension greater than 1, whether to intepret the batch as iid
        samples or batch of data points."""
        if self._x_is_iid is not None:
            return self._x_is_iid
        else:
            raise ValueError(
                "No observed data is available. Use `potential_fn.set_x(x_o)`."
            )

    def set_x(self, x_o: Optional[Tensor], interpret_as_iid: Optional[bool] = True):
        """Check the shape of the observed data and, if valid, set it."""
        if x_o is not None:
            x_o = process_x(x_o).to(self.device)
        self._x_o = x_o
        self._x_is_iid = interpret_as_iid

    @property
    def x_o(self) -> Tensor:
        """Return the observed data at which the potential is evaluated."""
        if self._x_o is not None:
            return self._x_o
        else:
            raise ValueError(
                "No observed data is available. Use `potential_fn.set_x(x_o)`."
            )

    @x_o.setter
    def x_o(self, x_o: Optional[Tensor]) -> None:
        """Check the shape of the observed data and, if valid, set it."""
        self.set_x(x_o)

    def return_x_o(self) -> Optional[Tensor]:
        """Return the observed data at which the potential is evaluated.

        Difference to the `x_o` property is that it will not raise an error if
        `self._x_o` is `None`.
        """
        return self._x_o


class CallablePotentialWrapper(BasePotential):
    """If `potential_fn` is a callable it gets wrapped as this."""

    allow_iid_x = True  # type: ignore

    def __init__(
        self,
        callable_potential,
        prior: Optional[Distribution],
        x_o: Optional[Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__(prior, x_o, device)
        self.callable_potential = callable_potential

    def __call__(self, theta, track_gradients: bool = True):
        with torch.set_grad_enabled(track_gradients):
            return self.callable_potential(theta=theta, x_o=self.x_o)
