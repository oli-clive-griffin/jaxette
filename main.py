from abc import ABC, abstractmethod

class HasValue(ABC):
    x: float

class Val(HasValue):
    def __init__(self, x):
        self.x = x

class Ops(Val):
    def __add__(self, other: HasValue):
        return AddTracer(self.x + other.x, children=[self, other])
    def __mul__(self, other: HasValue):
        return MulTracer(self.x * other.x, children=[self, other])

class Param(Ops):
    def __repr__(self):
        return f'Param({self.x})'

class Input(Ops):
    def __repr__(self):
        return f'Input({self.x})'

class Tracer:
    def __init__(self, x, *, children: list[HasValue]):
        self.x = x
        self.children = children
    
    @abstractmethod
    def grads(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self.x} {f", children={self.children}" if self.children is not None else ""})'

class MulTracer(Tracer, Ops):
    def grads(self, upstream):
        # nb: swapped 
        d_l = self.children[1].x * upstream
        d_r = self.children[0].x * upstream
        return [d_l, d_r]

class AddTracer(Tracer, Ops):
    def grads(self, upstream):
        return [upstream, upstream]

class d(ABC):
    d: float

class d_Param(d):
    def __init__(self, d: float):
        self.d = d

    def __repr__(self):
        return f"d_Param(d={self.d})"

class d_Input(d):
    def __init__(self, d: float):
        self.d = d

    def __repr__(self):
        return f"d_Param(d={self.d})"

class d_Tracer(d):
    def __init__(self, d: float, children: d):
        self.d = d
        self.children = children

    def __repr__(self):
        return f"d_Tracer(d={self.d}{f', children={self.children}' if self.children is not None else ''})"

def back_pass_tracer(node: Tracer | Param, upstream:float=1) -> d:
    if isinstance(node, Param):
        return d_Param(upstream)
    if isinstance(node, Input):
        return d_Input(upstream)
    if isinstance(node, Tracer):
        return d_Tracer(
            upstream,
            children=[back_pass_tracer(ch, d_ch) for ch, d_ch in zip(node.children, node.grads(upstream))]
        )
    raise TypeError(f"Unknown type {type(node)}")

def get_params(d_node: d, so_far:list[d]=[]) -> list[d_Param]:
    if isinstance(d_node, d_Param):
        return [*so_far, d_node]
    if isinstance(d_node, d_Input):
        return so_far
    if isinstance(d_node, d_Tracer):
        added = [param for child in d_node.children for param in get_params(child)]
        return [*so_far, *added]
    raise TypeError(f"Unknown type {type(d_node)}")

def grad(fn):
    def wrapper(*args, **kwargs):
        return get_params(back_pass_tracer(fn(*args, **kwargs)))
    return wrapper

if __name__ == "__main__":
    def model(params: list[float], input: float):
        return Param(params[0]) * Input(input) + Param(params[1])

    forward = grad(model)
    params = [2., 3.]
    grads = forward(params, 5.)
    print(grads)


