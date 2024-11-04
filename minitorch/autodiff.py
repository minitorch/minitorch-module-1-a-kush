from dataclasses import dataclass
from typing import Any, Iterable, Deque, Tuple, Dict
from collections import defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # [done]: Implement for Task 1.1.
    right_vals = list(vals)
    left_vals = list(vals)
    right_vals[arg] += epsilon
    left_vals[arg] -= epsilon
    return (f(*right_vals) - f(*left_vals)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # [done]: Implement for Task 1.4.
    def backward_dfs(cur: Variable, used: Dict[int, bool], traversal: Deque[Variable]):
        for input in cur.parents:
            if not used[input.unique_id]:
                used[input.unique_id] = True
                backward_dfs(input, used, traversal)
        traversal.appendleft(cur)
        return
    
    deque = Deque()
    backward_dfs(variable, defaultdict(bool), deque)
    return list(deque)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    derivs = defaultdict(float)
    derivs[variable.unique_id] = deriv

    top_sort = topological_sort(variable)

    for var in top_sort:
        if var.is_leaf():
            var.accumulate_derivative(derivs[var.unique_id])
        else:
            input_derivs = var.chain_rule(derivs[var.unique_id])
            for input, input_deriv in input_derivs:
                derivs[input.unique_id] += input_deriv
        
    
        

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
