# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Optional, Union

from torch import nn

from sbi.neural_nets.classifier import (
    build_linear_classifier,
    build_mlp_classifier,
    build_resnet_classifier,
)
from sbi.neural_nets.flow import (
    build_made,
    build_maf,
    build_maf_rqs,
    build_nsf,
    build_zuko_maf,
)
from sbi.neural_nets.mdn import build_mdn
from sbi.neural_nets.mnle import build_mnle
from sbi.neural_nets.vector_field import build_score_estimator

def classifier_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    embedding_net_theta: nn.Module = nn.Identity(),
    embedding_net_x: nn.Module = nn.Identity(),
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a classifier for learning density ratios.

    This function will usually be used for SNRE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Note that in the view of the SNRE classifier we build below, x=theta and y=x.

    Args:
        model: The type of classifier that will be created. One of [`linear`, `mlp`,
            `resnet`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        embedding_net_theta:  Optional embedding network for parameters $\theta$.
        embedding_net_x:  Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "embedding_net_x",
                "embedding_net_y",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                embedding_net_theta,
                embedding_net_x,
            ),
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model == "linear":
            return build_linear_classifier(
                batch_x=batch_theta, batch_y=batch_x, **kwargs
            )
        if model == "mlp":
            return build_mlp_classifier(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        if model == "resnet":
            return build_resnet_classifier(
                batch_x=batch_theta, batch_y=batch_x, **kwargs
            )
        else:
            raise NotImplementedError

    return build_fn


def likelihood_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the likelihood.

    This function will usually be used for SNLE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `maf_rqs`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for parameters $\theta$.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_x,
                z_score_theta,
                hidden_features,
                num_transforms,
                num_bins,
                embedding_net,
                num_components,
            ),
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model == "mdn":
            return build_mdn(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "made":
            return build_made(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "maf":
            return build_maf(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "maf_rqs":
            return build_maf_rqs(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "mnle":
            return build_mnle(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "zuko_maf":
            return build_zuko_maf(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        else:
            raise NotImplementedError

    return build_fn


def posterior_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs: Any,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the posterior.

    This function will usually be used for SNPE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `maf_rqs`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                embedding_net,
                num_components,
            ),
        ),
        **kwargs,
    )

    def build_fn_snpe_a(batch_theta, batch_x, num_components):
        """Build function for SNPE-A

        Extract the number of components from the kwargs, such that they are exposed as
        a kwargs, offering the possibility to later override this kwarg with
        `functools.partial`. This is necessary in order to make sure that the MDN in
        SNPE-A only has one component when running the Algorithm 1 part.
        """
        return build_mdn(
            batch_x=batch_theta,
            batch_y=batch_x,
            num_components=num_components,
            **kwargs,
        )

    def build_fn(batch_theta, batch_x):
        if model == "mdn":
            return build_mdn(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "made":
            return build_made(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf":
            return build_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf_rqs":
            return build_maf_rqs(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "zuko_maf":
            return build_zuko_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    if model == "mdn_snpe_a":
        if num_components != 10:
            raise ValueError(
                "You set `num_components`. For SNPE-A, this has to be done at "
                "instantiation of the inference object, i.e. "
                "`inference = SNPE_A(..., num_components=20)`"
            )
        kwargs.pop("num_components")

    return build_fn_snpe_a if model == "mdn_snpe_a" else build_fn
    

def posterior_score_nn(
    sde_type: str,
    score_net_type: Union[str, nn.Module] = "mlp",
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    t_embedding_dim: int = 16,
    hidden_features: int = 50,        
    embedding_net: nn.Module = nn.Identity(),
    **kwargs: Any,
) -> Callable:
    """Build util function that builds a ScoreEstimator object for score-based posteriors.

    Args:
        sde_type: SDE type used, which defines the mean and std functions. One of:
            - 'vp': Variance preserving.
            - 'subvp': Sub-variance preserving.
            - 've': Variance exploding.
            Defaults to 'vp'.
        score_net: Type of regression network. One of:
            - 'mlp': Fully connected feed-forward network.
            - 'resnet': Residual network (NOT IMPLEMENTED).
            -  nn.Module: Custom network
            Defaults to 'mlp'.                    
        z_score_theta: Whether to z-score thetas passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score xs passing into the network, same options as
            z_score_theta.
        t_embedding_dim: Embedding dimension of diffusion time. Defaults to 16.        
        hidden_features: Number of hidden units per layer. Defaults to 50.
        embedding_net: Embedding network for x (conditioning variable). Defaults to nn.Identity().

    Returns:
        Constructor function for NSPE.
    """
    
    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "sde_type",
                "score_net",
                "t_embedding_dim",
                "hidden_features",                
                "embedding_net_y",
                
            ),
            (
                z_score_theta,
                z_score_x,
                sde_type,
                score_net_type,
                t_embedding_dim,
                hidden_features,                
                embedding_net,
            ),
        ),
        **kwargs,
    )
    def build_fn(batch_theta, batch_x):
        """Build function wrapper for the build_score_estimator function that
        is required for the score posterior class.

        Args:
            batch_theta: a batch of theta.
            batch_x: a batch of x.

        Returns:
            Callable: a ScoreEstimator object.
        """
        return build_score_estimator(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        
    return build_fn