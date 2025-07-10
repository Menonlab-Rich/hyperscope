import ast
import math
import operator as op
import os
from pathlib import Path
from typing import Dict, List

from omegaconf import Container, DictConfig, ListConfig, OmegaConf

# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,  # Note: this is bitwise XOR, not power. Use ** for power.
    ast.USub: op.neg,
}


def eval_safe_expr(expr):
    """
    Safely evaluates a mathematical expression string using AST.
    Supports integers and floats.
    """
    # ast.parse returns a Module node for mode='eval', the expression is in .body
    evaled = _eval_node(ast.parse(expr, mode="eval").body)

    # Check if the float is close to its rounded integer value
    if isinstance(evaled, float) and math.isclose(evaled, round(evaled)):
        return int(round(evaled))
    return evaled


def _eval_node(node):
    """
    Recursive helper to evaluate AST nodes.
    """
    match node:
        case ast.Constant(value):  # No longer checking isinstance(value, int)
            # Allow both int and float constants
            if isinstance(value, (int, float)):
                return value
            else:
                # If you want to be stricter, you can raise an error for other constants
                raise TypeError(
                    f"Unsupported constant type: {type(value).__name__} for value {value}"
                )
        case ast.BinOp(left, op_type, right):
            # Evaluate left and right operands recursively
            left_val = _eval_node(left)
            right_val = _eval_node(right)

            # Look up the operator function and apply it
            if type(op_type) in operators:
                return operators[type(op_type)](left_val, right_val)
            else:
                raise TypeError(f"Unsupported binary operator: {type(op_type).__name__}")
        case ast.UnaryOp(op_type, operand):
            # Evaluate the operand recursively
            operand_val = _eval_node(operand)

            # Look up the unary operator function and apply it
            if type(op_type) in operators:
                return operators[type(op_type)](operand_val)
            else:
                raise TypeError(f"Unsupported unary operator: {type(op_type).__name__}")
        case _:
            # Raise an error for any unsupported AST node type
            raise TypeError(f"Unsupported AST node type: {type(node).__name__}")


OmegaConf.register_new_resolver("path", lambda path: Path(os.path.expanduser(path)))

OmegaConf.register_new_resolver("math", eval_safe_expr)


def _get_kwargs_from_cfg(cfg: Container, key_mappings: Dict[str, str]) -> Dict:
    """
    Selects keys from an OmegaConf container and maps them to a new dict.

    Args:
        cfg: The OmegaConf container.
        key_mappings: A dictionary mapping {target_kwarg_name: source_config_path}.

    Returns:
        A dictionary with the populated keyword arguments.
    """
    kwargs = {}
    for kwarg_name, config_path in key_mappings.items():
        value = OmegaConf.select(cfg, config_path, default=None)
        if value is not None:
            kwargs[kwarg_name] = value
    return kwargs


class Config(DictConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __contains__(self, key):
        if type(key) is str:
            return OmegaConf.select(self, key, default=None) is not None
        else:
            return key in self

    def select_as_kwargs(self, mappings):
        return _get_kwargs_from_cfg(self, mappings)
