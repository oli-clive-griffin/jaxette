from abc import ABC, abstractmethod

class Ops:
    def __init__(self, x):
        self.x = x
    def __add__(self, other: 'Ops'):
        return AddTracer(self.x + other.x, children=[self, other])
    def __mul__(self, other: 'Ops'):
        return MulTracer(self.x * other.x, children=[self, other])

class Param(Ops):
    def __repr__(self):
        return f'Param({self.x})'

class Input(Ops):
    def __repr__(self):
        return f'Input({self.x})'

class Tracer(ABC, Ops):
    def __init__(self, x, *, children: list[Ops]):
        self.x = x
        self.children = children
    
    @abstractmethod
    def grads(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self.x} {f", children={self.children}" if self.children is not None else ""})'

class MulTracer(Tracer):
    def grads(self, upstream):
        # nb: swapped 
        d_l = self.children[1].x * upstream
        d_r = self.children[0].x * upstream
        return [d_l, d_r]

class AddTracer(Tracer):
    def grads(self, upstream):
        return [upstream, upstream]

def back_pass_tracer(node: Tracer | Param | Input, upstream:float=1):
    if isinstance(node, Input):
        return []
    if isinstance(node, Param):
        return [upstream]
    if isinstance(node, Tracer):
        return [
            param
            for ch, d_ch in zip(node.children, node.grads(upstream))
            for param in back_pass_tracer(ch, d_ch)
        ]
    raise TypeError(f"Unknown type {type(node)}")

def grad(fn):
    def wrapper(*args, **kwargs):
        return back_pass_tracer(fn(*args, **kwargs))
    return wrapper

if __name__ == "__main__":
    def model(params: list[float], input: float):
        return Param(params[0]) * Input(input) + Param(params[1])

    forward = grad(model)
    params = [2., 3.]
    grads = forward(params, 5.)
    print(grads)


