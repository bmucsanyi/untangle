"""Model surgery utilities."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module
from torch.nn.utils import parametrize


@dataclass
class ModuleData:
    """Module data container."""

    variable_name: str
    module_name: str
    module: Module


def deep_setattr(obj: object, attr_path: str, value: object) -> None:
    """Sets a deeply nested attribute on an object."""
    parts = attr_path.split(".")

    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)

    if parts[-1].isdigit():
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)


def replace(model: Module, source_regex: str, target_module: Module) -> None:
    """Replaces modules in the model that match the regex with the module."""
    source_regex = re.compile(source_regex)

    module_data_list = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_data_list = [
        module_data
        for module_data in module_data_list
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_data_list:
        deep_setattr(
            model,
            matched_module_data.variable_name,
            target_module(matched_module_data.module),
        )


def replace_cond(
    model: Module,
    source_regex: str,
    cond: Callable[[Module], bool],
    target_module_true: Module,
    target_module_false: Module,
) -> None:
    """Conditionally replaces modules in the model based on a condition."""
    source_regex = re.compile(source_regex)

    module_data_list = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_data_list = [
        module_data
        for module_data in module_data_list
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_data_list:
        target_module = (
            target_module_true
            if cond(matched_module_data.module)
            else target_module_false
        )
        deep_setattr(
            model,
            matched_module_data.variable_name,
            target_module(matched_module_data.module),
        )


def register(
    model: Module,
    source_regex: str,
    attribute_name: str,
    target_parametrization: Module,
) -> None:
    """Registers a parametrization for modules in the model that match the regex."""
    source_regex = re.compile(source_regex)

    module_data_list = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_data_list = [
        module_data
        for module_data in module_data_list
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_data_list:
        module = matched_module_data.module
        module_name = matched_module_data.module_name
        weight = getattr(module, attribute_name, None)
        if not isinstance(weight, Tensor):
            msg = (
                f"Module '{module_name}' has no parameter or buffer with name "
                f"'{attribute_name}'"
            )
            raise TypeError(msg)

        parametrize.register_parametrization(
            module, attribute_name, target_parametrization(module=module), unsafe=True
        )


def register_cond(
    model: Module,
    source_regex: str,
    attribute_name: str,
    cond: Callable[[Module], bool],
    target_parametrization_true: Module,
    target_parametrization_false: Module,
) -> None:
    """Conditionally registers a parametrization for modules in the model."""
    compiled_source_regex = re.compile(source_regex)

    module_data_list = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_data_list = [
        module_data
        for module_data in module_data_list
        if compiled_source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_data_list:
        module = matched_module_data.module
        module_name = matched_module_data.module_name
        weight = getattr(module, attribute_name, None)
        if not isinstance(weight, Tensor):
            msg = (
                f"Module '{module_name}' has no parameter or buffer with name "
                f"'{attribute_name}'"
            )
            raise TypeError(msg)

        target_parametrization = (
            target_parametrization_true
            if cond(matched_module_data.module)
            else target_parametrization_false
        )

        parametrize.register_parametrization(
            module, attribute_name, target_parametrization(module=module), unsafe=True
        )
