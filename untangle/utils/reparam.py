"""ReLU reparametrization utilities."""

import math
from copy import deepcopy
from functools import partial
from math import prod

import torch
from torchmin import minimize


class ShapeCapture:
    """Captures shapes of input and output tensors of each layer in a model."""

    def __init__(self, model):
        self.model = model
        self.intermediates = []

    def hook_fn(self, module, input, output, name):
        del module
        self.intermediates.append({name: (input[0].shape, output.shape)})

    def __enter__(self):
        self.hooks = [
            layer.register_forward_hook(partial(self.hook_fn, name=name))
            for name, layer in self.model.named_children()
        ]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        for hook in self.hooks:
            hook.remove()


def predict_with_shapes(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, list]:
    """Extracts shapes from the forward pass."""
    with ShapeCapture(model) as sc:
        y = model(x)
        return y, sc.intermediates


def flatten_shapes(shapes: list) -> list:
    """Flatten shapes."""
    return [
        (k, (math.prod(v[0]), math.prod(v[1])))
        for shape in shapes
        for k, v in shape.items()
    ]


def augment_shapes(
    model: torch.nn.Module,
    params: dict,
    shapes: list,
    input_shape: tuple,
) -> dict:
    # Flatten shapes and add input shape
    shapes_flat = flatten_shapes(shapes)
    shapes_flat.insert(0, ("input", math.prod(input_shape)))
    # Augment modules with shapes
    for i, (n, _) in enumerate(model.named_children(), start=1):
        layer_name, curr_shape = shapes_flat[i]
        if layer_name != n:
            msg = f"Expected {n}, got {layer_name}, possibly double use of layer name."
            raise ValueError(msg)
        if (param_name := (layer_name + ".weight")) in params:
            params[param_name].shapes = {"in": curr_shape[0], "out": curr_shape[1]}
    return params


def add_shapes_to_parameters(
    model: torch.nn.Module, params: dict, input_shape: tuple
) -> dict:
    _, shapes = predict_with_shapes(model, torch.randn(*input_shape))
    return augment_shapes(model, params, shapes, input_shape)


def deepcopy_and_add_shapes_to_parameters(
    params: dict, model: torch.nn.Module, input_shape: tuple
) -> dict:
    return add_shapes_to_parameters(
        model=model, params=deepcopy(params), input_shape=input_shape
    )


def get_parameter_tree(params: dict, *, conv_as_dense: bool) -> dict:
    ndict = {}
    # Extract layers and parameters
    for n, p in params.items():
        nsplit = n.split(".")
        layer_name = "".join(*nsplit[:-1])
        ndict[layer_name] = {nsplit[-1]: p, **ndict.get(layer_name, {})}
        if "weight" in n:
            ndict[layer_name]["weight_shape"] = (p.shape[0], prod(p.shape[1:]))
            if conv_as_dense:
                if not hasattr(p, "shapes"):
                    msg = (
                        "shapes not found in params dictionary, augment "
                        "dictionary using deepcopy_and_attach_shape."
                    )
                    raise ValueError(msg)
                ndict[layer_name]["out_in_dim"] = (p.shapes["out"], p.shapes["in"])
            else:
                ndict[layer_name]["out_in_dim"] = ndict[layer_name]["weight_shape"]
    return ndict


def from_parameter_tree(params_tree: dict) -> dict:
    nparams = {}
    for layer_name, layer_params in params_tree.items():
        for n, p in layer_params.items():
            if n not in {"out_in_dim", "conv_as_dense", "weight_shape"}:
                nparams[layer_name + "." + n] = p
    return nparams


def left_action(obj: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # (C', C') * (C', C, K1, K2)
    return torch.einsum("ij,j...->i...", action, obj) if action is not None else obj


def right_action(obj: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # (C', C') * (C'', C', K1', K2')
    return torch.einsum("ij,kj...->ki...", action, obj) if action is not None else obj


def get_reparam_distance(G: list[torch.Tensor]) -> float:
    G = [g for g in G if g is not None]
    return torch.linalg.norm(torch.log(torch.concatenate(G)))


def get_layer_scaling(
    out_in_dims: tuple, layer_one_dims: tuple, *, layer_scaling: list
) -> torch.Tensor:
    if layer_scaling == "uniform":  # Reparam.UNIFORM:
        scaling = 1
    elif layer_scaling == "layer_out":  # Reparam.LAYER_OUT:
        scaling = out_in_dims[0] / layer_one_dims[0]  # For numerical stability
    elif layer_scaling == "layer_in":  # Reparam.LAYER_IN:
        scaling = out_in_dims[1] / layer_one_dims[1]  # For numerical stability
    elif layer_scaling == "layer_size":  # Reparam.LAYER_SIZE:
        scaling = (
            out_in_dims[0] * out_in_dims[1] / (layer_one_dims[0] * layer_one_dims[1])
        )  # For numerical stability
    else:
        msg = "invalid Reparam provided to g_norm"
        raise ValueError(msg)

    return torch.tensor(scaling, dtype=torch.float32)


def apply_reparam(G_action: list[torch.Tensor], params_tree: dict) -> dict:
    D_inv = None
    n_layers = len(params_tree)

    # Extend by None if necessary
    G_action = [*G_action, None] if len(G_action) < n_layers else G_action
    if len(G_action) != n_layers:
        msg = "length of G does not match number of layers"
        raise ValueError(msg)

    for i, (layer_name, layer) in enumerate(list(params_tree.items())):
        last_layer = i == n_layers - 1
        G = G_action[i]
        D = torch.diag(G) if not last_layer else None
        params_tree[layer_name]["weight"] = left_action(
            right_action(layer["weight"], action=D_inv), action=D
        )
        params_tree[layer_name]["bias"] = (
            D @ layer["bias"] if not last_layer else layer["bias"]
        )
        D_inv = torch.diag(1 / G) if not last_layer else None

    return params_tree


def get_g_norm(params_tree: dict, *, layer_scaling: list) -> tuple[list, list]:
    _scaling = {}
    G_action = []
    n_layers = len(params_tree)
    for i, (layer_name, layer) in enumerate(list(params_tree.items())):
        if "fc" in layer_name or "conv" in layer_name:
            out_in_dim = layer["out_in_dim"]
            if i == 0:
                layer_one_dim = out_in_dim
            layer_weight = right_action(
                layer["weight"], action=torch.diag(1 / G_action[-1]) if i > 0 else None
            ).reshape(layer["weight_shape"])
            norms = torch.linalg.norm(layer_weight, axis=-1)
            G_action.append(
                1 / (norms * math.sqrt(out_in_dim[0])) if i < (n_layers - 1) else None
            )
            _scaling[layer_name] = get_layer_scaling(
                out_in_dim, layer_one_dim, layer_scaling=layer_scaling
            )

    return G_action, _scaling


def get_one_action(params_tree: dict) -> list:
    return [
        torch.ones(layer["weight_shape"][0])
        for layer in list(params_tree.values())[:-1]
    ]


def get_zero_action(params_tree: dict) -> list:
    return [
        torch.zeros(layer["weight_shape"][0])
        for layer in list(params_tree.values())[:-1]
    ]


def get_indices(params_tree: dict) -> list:
    return [layer["weight_shape"][0] for layer in list(params_tree.values())[:-1]]


def calculate_layer_scaling(params_tree: dict, layer_scaling: list) -> list:
    one_action = get_one_action(params_tree)

    # Weight norm last layer
    list_params = list(params_tree.items())
    list_params.reverse()
    for layer_name, layer in list_params:
        if "fc" in layer_name or "conv" in layer_name:
            out_in_dim = layer["out_in_dim"]
            normWL = torch.linalg.norm(layer["weight"]) * math.sqrt(
                out_in_dim[0] / layer["weight"].shape[0]
            )
            break

    n_layers = len(layer_scaling)
    ls_prod = prod(layer_scaling.values())

    # Calculate alpha
    alphas = [l * (normWL / ls_prod) ** (1 / n_layers) for l in layer_scaling.values()]
    alphas[-1] /= normWL

    # Calculate beta
    betas = torch.cumprod(torch.tensor(alphas), dim=0)

    return [beta * one for beta, one in zip(betas, one_action, strict=False)]


def multiply_g(G_action1, G_action2) -> list:
    # Extend by None if necessary
    G_action1 = [*G_action1, None] if G_action1[-1] is not None else G_action1
    G_action2 = [*G_action2, None] if G_action2[-1] is not None else G_action2
    return [G_action1[l] * G_action2[l] for l in range(len(G_action1) - 1)]


def g_norm(
    params: dict, *, conv_as_dense: bool, layer_scaling: str
) -> tuple[dict, float]:
    # Extract weights
    params_tree = get_parameter_tree(params, conv_as_dense=conv_as_dense)

    # Calculate weight norm
    G_action, _scaling = get_g_norm(params_tree, layer_scaling=layer_scaling)

    # Apply G_action
    params_tree = apply_reparam(G_action, params_tree)

    # Calculate layer scaling action
    G_action_scaling = calculate_layer_scaling(params_tree, _scaling)

    # Apply G_action_scaling
    params_tree = apply_reparam(G_action_scaling, params_tree)

    # Form new parameters and calculate distance
    new_params = from_parameter_tree(params_tree)
    G_action_total = multiply_g(G_action, G_action_scaling)
    dist_reparam = get_reparam_distance(G_action_total)

    return new_params, dist_reparam


def get_params_norm(params_tree: dict) -> float:
    norm = torch.tensor(0.0)
    for layer in params_tree.values():
        norm += (
            (torch.sum(layer["weight"] ** 2) + torch.sum(layer["bias"] ** 2))
            * layer["out_in_dim"][0]
            / layer["weight"].shape[0]
        )
    return torch.sqrt(norm)


def min_norm_objective(logG: list, params_tree: dict) -> float:
    # G_action to list of layer actions
    indices = get_indices(params_tree)
    G_action = [torch.exp(g) for g in logG.split_with_sizes(indices)]
    # Apply reparametrization
    params_tree = apply_reparam(G_action, params_tree)
    # Return norm of new parameters
    return get_params_norm(params_tree)


def g_min_norm(params: dict, *, conv_as_dense: bool) -> tuple[dict, float]:
    # Extract weights
    params_tree = get_parameter_tree(params, conv_as_dense=conv_as_dense)

    # Calculate G_min_norm
    obj = minimize(
        partial(min_norm_objective, params_tree=params_tree),
        torch.concat(get_zero_action(params_tree)),
        method="newton-exact",
    )

    # Apply reparametrization
    indices = get_indices(params_tree)
    G_action = [torch.exp(g) for g in obj.x.split_with_sizes(indices)]
    params_tree = apply_reparam(
        G_action, get_parameter_tree(params, conv_as_dense=conv_as_dense)
    )

    # Form new parameters and calculate distance
    new_params = from_parameter_tree(params_tree)
    dist_reparam = get_reparam_distance(G_action)

    return new_params, dist_reparam


def g_rand(params: dict, *, conv_as_dense: bool) -> tuple[dict, float]:
    # Extract weights
    params_tree = get_parameter_tree(params, conv_as_dense=conv_as_dense)

    # Draw random action
    indices = get_indices(params_tree)
    rand_action = torch.distributions.log_normal.LogNormal(0, 1).sample(
        torch.concat(get_one_action(params_tree)).shape
    )
    rand_action = list(rand_action.split_with_sizes(indices))

    # Apply reparametrization
    params_tree = apply_reparam(rand_action, params_tree)

    # Form new parameters and calculate distance
    new_params = from_parameter_tree(params_tree)
    dist_reparam = get_reparam_distance(rand_action)

    return new_params, dist_reparam
