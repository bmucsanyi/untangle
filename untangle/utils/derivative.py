"""Model derivative utils."""

import torch


def jvp(y, x, v, *, is_grads_batched=False, retain_graph=False):
    """Multiplies the vector `v` with the Jacobian of `y` w.r.t. `x`.

    Corresponds to the application of the R-operator.

    Parameters:
    -----------
        y: torch.Tensor or [torch.Tensor]
            Outputs of the differentiated function.
        x: torch.Tensor or [torch.Tensor]
            Inputs w.r.t. which the gradient will be returned.
        v: torch.Tensor or [torch.Tensor]
            The vector to be multiplied by the Jacobian.
        retain_graph: bool
            Whether to retain the JVP graph.
    """
    if isinstance(y, tuple):
        w = [torch.zeros_like(y).requires_grad_(True) for y in y]
    else:
        w = torch.zeros_like(y).requires_grad_(True)

    gs = torch.autograd.grad(
        y,
        x,
        grad_outputs=w,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    re = torch.autograd.grad(
        gs,
        w,
        grad_outputs=v,
        is_grads_batched=is_grads_batched,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    return tuple(j.detach() for j in re)


def vjp(y, x, v, *, is_grads_batched=False, retain_graph=False):
    """Multiplies the vector `v` with the (transposed) Jacobian of `f` w.r.t. `x`.

    Corresponds to the application of the L-operator.

    Parameters:
    -----------
        y: torch.Tensor or [torch.Tensor]
            Outputs of the differentiated function.
        x: torch.Tensor or [torch.Tensor]
            Inputs w.r.t. which the gradient will be returned.
        v: torch.Tensor or [torch.Tensor]
            The vector to be multiplied by the transposed Jacobian.
        retain_graph: bool
            Whether to retain the VJP graph.
    """
    vJ = torch.autograd.grad(
        y,
        x,
        grad_outputs=v,
        retain_graph=retain_graph,
        allow_unused=True,
        is_grads_batched=is_grads_batched,
    )

    return tuple(j.detach() for j in vJ)
