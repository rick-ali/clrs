import haiku as hk
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional
from jax import lax
import numpy as np
import torch

class SemiringLayer(hk.Module):
    """Linear module."""

    def __init__(
        self,
        output_size: int,
        basis: Optional[float] = None,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = 'logsumexp_semiring_layer',
    ):
        """Constructs the logsumexp_semiring_layer module.

        Args:
        output_size: Output dimensionality.
        with_bias: Whether to add a bias to the output.
        w_init: Optional initializer for weights. By default, uses random values
            from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
            https://arxiv.org/abs/1502.03167v3.
        b_init: Optional initializer for bias. By default, zero.
        name: Name of the module.
        """
        super().__init__(name=name)
        self.input_size = None
        self.basis = basis
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def __call__(
        self,
        inputs: jax.Array,
        *,
        precision: Optional[lax.Precision] = None,
    ) -> jax.Array:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        if self.basis is None:  # Default basis 
            max_inputs, max_w = np.max(inputs), np.max(w)
            exp_inputs, exp_w = inputs - max_inputs, w - max_w
            exp_inputs = jnp.exp(exp_inputs)
            exp_w = jnp.exp(exp_w)
            out = jnp.dot(exp_inputs, exp_w)
            out = jnp.log(out)
            out += max_inputs + max_w
            #out = jnp.log(jnp.dot(jnp.exp(inputs), jnp.exp(w), precision=precision))
        else:
            max_inputs, max_w = np.max(inputs), np.max(w)
            exp_inputs, exp_w = inputs - max_inputs, w - max_w
            exp_inputs = self.basis**exp_inputs
            exp_w = self.basis**exp_w
            out = jnp.dot(exp_inputs, exp_w)
            out = jnp.log(out)/jnp.log(self.basis)
            out += max_inputs + max_w
            #out = jnp.log(jnp.dot(self.basis**(inputs), self.basis**(w), precision=precision))/jnp.log(self.basis)
        
        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out

class OtherSideLinear(hk.Module):
  """Multiply matrix on the other side module."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs the Linear module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(
      self,
      inputs: jax.Array,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jax.Array:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    out = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out