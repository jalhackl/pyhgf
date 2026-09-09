# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges

#
from typing import Dict
from typing import TYPE_CHECKING, List, Tuple
from pyhgf.typing import Sequence

import jax.numpy as jnp
from jax import grad, jit
from jax.lax import cond
from jax.tree_util import Partial
from jax import lax



@partial(jit, static_argnames=("edges", "node_idx"))
def predict_mean(
    attributes: dict,
    edges: Edges,
    node_idx: int,
) -> Array:
    r"""Compute the expected mean of a continuous state node.

    The expected mean at time :math:`k` for a state node :math:`a` with optional value
    parent(s) :math:`b` is in [1]_ given by:

    .. math::

        \hat{\mu}_a^{(k)} = \lambda_a \mu_a^{(k-1)} + P_a^{(k)}

    where :math:`P_a^{(k)}` is the drift rate (the total predicted drift of the expected
    mean, which sums the tonic and - optionally - phasic drifts). The variable
    :math:`lambda_a` represents the state's autoconnection strength, with
    :math:`\lambda_a \in [0, 1]`. When :math:`\lambda_a = 1`, the node is performing a
    Gaussian Random Walk using the value :math:`P_a^{(k)}` as total drift rate. When
    :math:`\lambda_a < 1`, the state will revert back to the total mean :math:`M_a`,
    which is given by:

    .. math::
            M_a = \frac{\rho_a + f\left(\hat{\mu}_b^{(k)}\right)} {1-\lambda_a},

    If :math:`\lambda_a = 0`, the node is not influenced by its own mean anymore, but
    by the value received by the value parent.

    .. hint::

        By combining one parameter :math:`\lambda_a \in [0, 1]` and the influence of
        value parents, it is possible to implement both Gaussian Random Walks and
        Autoregressive Processes, without requiring specific coupling types.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_mean :
        The new expected mean of the state node.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    time_step = attributes[-1]["time_step"]

    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = attributes[node_idx]["tonic_drift"]

    # Look at the (optional) value parents for this node
    # and update the drift rate accordingly
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            # look at each value parent
            # and get the coupling function to compute the drift
            child_position = edges[value_parent_idx].value_children.index(node_idx)
            coupling_fn = edges[value_parent_idx].coupling_fn[child_position]
            if coupling_fn is None:
                parent_value = attributes[value_parent_idx]["expected_mean"]
            else:
                parent_value = coupling_fn(
                    attributes[value_parent_idx]["expected_mean"]
                )

            driftrate += psi * parent_value

    # The new expected mean from the previous value
    expected_mean = (
        attributes[node_idx]["autoconnection_strength"] * attributes[node_idx]["mean"]
    ) + (time_step * driftrate)

    return expected_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision(
    attributes: dict, edges: Edges, node_idx: int
) -> tuple[Array, Array]:
    r"""Compute the expected precision of a continuous state node.

    The expected precision at time :math:`k` for a state node :math:`a` is given by
    [1]_:

    .. math::

        \hat{\pi}_a^{(k)} = \frac{1}{\frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)}}

    where :math:`\Omega_a^{(k)}` is the *total predicted volatility*. This term is the
    sum of the tonic (endogenous) and phasic (exogenous) volatility, such as:

    .. math::

        \Omega_a^{(k)} = t^{(k)}
        \exp{ \left( \omega_a + \sum_{j=1}^{N_{vopa}} \kappa_j \hat{\mu}_a^{(k-1)} \right) }


    with :math:`\kappa_j` the volatility coupling strength with the volatility parent
    :math:`j`.

    The *effective precision* :math:`\gamma_a^{(k)}` is given by:

    .. math::

        \gamma_a^{(k)} = \Omega_a^{(k)} \hat{\pi}_a^{(k)}

    This value is also saved in the node for later use during the update steps.


    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_precision :
        The new expected precision of the value parent.
    effective_precision :
        The effective_precision :math:`\gamma_a^{(k)}`. This value is stored in the
        node for later use in the update steps.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    time_step = attributes[-1]["time_step"]

    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling
                * attributes[volatility_parents_idx]["expected_mean"]
            )

    # compute the predicted_volatility from the total volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Estimate the new expected precision for the node
    expected_precision = 1 / (
        (1 / attributes[node_idx]["precision"]) + predicted_volatility
    )

    # compute the effective precision (γ)
    effective_precision = predicted_volatility * expected_precision

    return expected_precision, effective_precision


#standard function
def continuous_node_prediction_default(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a continuous node [1]_.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.

    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # if this node has volatility parent(s), store the current variance
    # to be used by the posterior update if using unbounded approximation
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # Get the new expected mean
    expected_mean = predict_mean(attributes, edges, node_idx)

    # Get the new expected precision and predicted volatility (Ω)
    expected_precision, effective_precision = predict_precision(
        attributes, edges, node_idx
    )

    # Update this node's parameters

    # 1. input node without volatility parent
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]

    # 2. regular continuous state node, or input with volatility parent
    else:
        attributes[node_idx]["expected_precision"] = expected_precision

    attributes[node_idx]["temp"]["effective_precision"] = effective_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes


# new: adding 'forget' functionality for multiple, partially masked inputs
@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction(
    attributes: dict, node_idx: int, edges: Edges, **args
) -> dict:
    """Update the expected mean and expected precision of a continuous node [1]_.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.

    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # if this node has volatility parent(s), store the current variance
    # to be used by the posterior update if using unbounded approximation
    attributes[node_idx]["temp"]["current_variance"] = (
        1 / attributes[node_idx]["precision"]
    )

    # Get the new expected mean
    expected_mean = predict_mean(attributes, edges, node_idx)

    # Get the new expected precision and predicted volatility (Ω)
    expected_precision, effective_precision = predict_precision(
        attributes, edges, node_idx
    )
    
    ###################################################
    #NEW - mode 4 is working fine, 0 (none) is default
    mode_map = {"none": 0, "hgf_decay": 1, "regret_decay": 2, "regret_additive_precision": 3, "regret_additive_mult": 4}


    default_observed_from  = {
    2: 0,   # when updating node 2, check node 0
    3: 1,   # when updating node 3, check node 1
    }

    default_regret_from    = {
    2: 1,   # when updating node 2, check node 1
    3: 0,   # when updating node 3, check node 0
    }
    
    # Allow override from global config node (attributes[-1])
    observed_from = attributes[-1].get("observed_from", default_observed_from)
    regret_from   = attributes[-1].get("regret_from",   default_regret_from)
    
    # which nodes observed value to use
    source_idx = observed_from.get(node_idx, node_idx)

    # JAX: scalar boolean
    observed = attributes[source_idx]["observed"]
    cond_observed = (observed == 0)

    # Extract mean_forget if it exists, otherwise 0 (JAX-friendly)
    mean_forget = attributes[node_idx].get("mean_forget", 0.0)
    mean_forgetneg = attributes[node_idx].get("mean_forgetneg", 0.0)
    
    #precision = attributes[node_idx]["precision"]
    
    

    precision = attributes[node_idx]["precision"] - expected_precision
    precision_val = attributes[node_idx]["precision"]
    
    
    if "forget" in attributes[node_idx]:
        time_multiplier = attributes[node_idx]["forget"]
    else:
        time_multiplier = 1
        

    # mean for differentiating between mean_forget and mean_forgetnet
    source_mean = attributes[source_idx]["mean"]

    source_regret_idx = regret_from.get(node_idx, node_idx)
    source_regret_mean = attributes[source_regret_idx]["mean"]
    # for the binary input, the 'mean' is either 1 (rewarded) or 0 (not rewarded)
    
    source_regret_precision = attributes[source_regret_idx]["precision"]

    
    if "precision_full" in attributes[-1]:
        precision = precision_val
    
    precision_for_forgetting = precision
    if "use_regret_precision" in attributes[-1]:
        print("use_regret_precision")
        precision_for_forgetting = source_regret_precision * attributes[-1]["use_regret_precision"] + precision * (1 - attributes[-1]["use_regret_precision"])

    if "precision_full" in attributes[-1]:
        precision_for_forgetting = precision_val

    if "precision_other_weight" in attributes[node_idx]:
        precision_other_weight = attributes[node_idx]["precision_other_weight"]
    else:
        precision_other_weight = 1
    

    if "use_effective_precision" in attributes[-1]:
        print("use effective precision")
        #precision_for_forgetting = source_regret_precision - precision_other_weight * source_regret_precision 
        precision_for_forgetting = source_regret_precision / (1 + precision_other_weight * source_regret_precision)



    def no_forgetting(pm):
        
        return pm


    def hgf_decay(pm):
        
        return pm * jnp.exp(-mean_forget / precision_for_forgetting)


    def regret_decay(pm):
        
        rate = (
            mean_forget     * source_regret_mean
            + mean_forgetneg * (1 - source_regret_mean)
        )
        return pm * jnp.exp(-rate / precision_for_forgetting)


    def regret_additive_precision(pm):
        
        return (
            pm
            - (mean_forget     * source_regret_mean) / precision_for_forgetting
            - (mean_forgetneg * (1 - source_regret_mean)) / precision_for_forgetting
        )


    def regret_additive_mult(pm):
        
        return (
            pm
            # in case of NO reward, this term is zero 
            - mean_forget     * precision_for_forgetting * source_regret_mean
            # in case of reward, this term is zero 
            - mean_forgetneg * precision_for_forgetting * (1 - source_regret_mean)
        )

    mode_idx = attributes[-1].get("forget_mode", 0)
    
    def apply_forgetting(pm):
        return lax.switch(
            mode_idx,
            [
                no_forgetting,                # 0
                hgf_decay,                    # 1
                regret_decay,                 # 2
                regret_additive_precision,    # 3
                regret_additive_mult,         # 4
            ],
            pm,   
        )


    # lax.cond necessary instead of standard if-clause
    def keep(pm):
        return pm

    expected_mean = lax.cond(
        cond_observed,
        apply_forgetting,
        keep,
        expected_mean,
    )

    
    #############################################
    
    
    

    # Update this node's parameters

    # 1. input node without volatility parent
    if (
        (edges[node_idx].value_children is None)
        and (edges[node_idx].volatility_children is None)
        and (edges[node_idx].volatility_parents is None)
    ):
        attributes[node_idx]["expected_precision"] = attributes[node_idx]["precision"]

    # 2. regular continuous state node, or input with volatility parent
    else:
        attributes[node_idx]["expected_precision"] = expected_precision

    attributes[node_idx]["temp"]["effective_precision"] = effective_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes
