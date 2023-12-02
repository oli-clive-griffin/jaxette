from abc import ABC, abstractmethod
import math

def autocast_to_param(fn):
    def wrapper(*args, **kwargs):
        args = [Param(arg) if isinstance(arg, (float, int)) else arg for arg in args ]
        kwargs = { k: Param(v) if isinstance(v, (float, int)) else v for k, v in kwargs.items() }
        return fn(*args, **kwargs)
    return wrapper

class Ops:
    def __init__(self, x: float):
        self.x = x
    @autocast_to_param
    def __add__(self, other: 'Ops'): return AddTracer.forward(self, other)
    @autocast_to_param
    def __sub__(self, other: 'Ops'): return self + -other
    @autocast_to_param
    def __mul__(self, other: 'Ops'): return MulTracer.forward(self, other)
    @autocast_to_param
    def __neg__(self): return self * -1
    @autocast_to_param
    def __pow__(self, other: 'Ops'): raise NotImplementedError() # return PowTracer.forward(self, other)
    @autocast_to_param
    def __truediv__(self, other: 'Ops'): return DivTracer.forward(self, other)


class Param(Ops):
    def __repr__(self):
        return f'Param({self.x})'

class Input(Ops):
    def __repr__(self):
        return f'Input({self.x})'

class Tracer(ABC, Ops):
    @abstractmethod
    def forward(*args: Ops):
        raise NotImplementedError

    def __init__(self, x, *, children: list[Ops]):
        self.x = x
        self.children = children
    
    @abstractmethod
    def grads(self) -> list[float]:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self.x} {f", children={self.children}" if self.children is not None else ""})'

class MulTracer(Tracer):
    @staticmethod
    def forward(a: Ops, b: Ops):
        return MulTracer(a.x * b.x, children=[a, b])

    def grads(self, upstream):
        # nb: swapped 
        d_l = self.children[1].x * upstream
        d_r = self.children[0].x * upstream
        return [d_l, d_r]

class AddTracer(Tracer):
    @staticmethod
    def forward(a: Ops, b: Ops):
        return AddTracer(a.x + b.x, children=[a, b])

    def grads(self, upstream):
        return [upstream, upstream]

class DivTracer(Tracer):
    @staticmethod
    def forward(a: Ops, b: Ops):
        return DivTracer(a.x / b.x, children=[a, b])
    
    def grads(self):
        a, b = self.children
        return [1/b, -a/b**2]


class SinTracer(Tracer):
    @staticmethod
    def forward(x: Ops):
        return SinTracer(math.sin(x.x), children=[x])

    def grads(self, upstream):
        return [upstream * math.cos(self.x)]

def sin(x):
    return SinTracer.forward(x)

class CosTracer(Tracer):
    @staticmethod
    def forward(x: Ops):
        return CosTracer(math.cos(x.x), children=[x])

    def grads(self, upstream):
        return [-upstream * math.sin(self.x)]

def cos(x):
    return CosTracer.forward(x)

class SquareTracer(Tracer):
    @staticmethod
    def forward(x: Ops):
        return SquareTracer(x.x**2, children=[x])

    def grads(self, upstream):
        return [upstream * 2 * self.x]
    
def square(x):
    return SquareTracer.forward(x)

# not implementing for now because of the need to handle negative numbers in log
# class PowTracer(Tracer):
#     @staticmethod
#     def forward(x: Ops, n: Ops):
#         if x.x < 0:
#             raise ValueError("x must be positive")

#         return PowTracer(x.x**n.x, children=[x, n])

#     def grads(self, upstream):
#         x, n = (ch.x for ch in self.childrenj)
#         # ```mathjax
#         # \frac{d}{dx} x^n = n x^{n-1}
#         # \frac{d}{dn} x^n = x^n \log(x)
#         # ```
#         try:
#             return [upstream * n * x**(n-1), upstream * x**n * math.log(x)]
#         except:
#             breakpoint()

def back_pass_tracer(node: Tracer | Param | Input, upstream:float=1):
    if isinstance(node, Input):
        return []
    if isinstance(node, Param):
        return [upstream]
    if isinstance(node, Tracer):
        grads = node.grads(upstream)
        if len(node.children) != len(grads):
            raise ValueError(f"Expected {len(node.children)} grads, got {len(grads)}")
        return [
            param
            for ch, d_ch in zip(node.children, grads)
            for param in back_pass_tracer(ch, d_ch)
        ]
    raise TypeError(f"Unknown type {type(node)}")

def grad(fn):
    def wrapper(*args, **kwargs):
        return back_pass_tracer(fn(*args, **kwargs))
    return wrapper

if __name__ == "__main__":
    lr = 0.001

    def model(params: list[float], x: float) -> Tracer:
        p_1, p_2 = params
        return square(Param(p_1)) * square(Param(p_2)) - Input(x)

    def forward(params: list[float], x: float, y: float):
        x = model(params, x)
        return square(x - y)

    def update(params: list[float], grads: list[float]):
        return [p - lr * g for p, g in zip(params, grads)]

    model_grad = grad(model)

    params = [2., 3.]
    for _ in range(1000):
        grads = model_grad(params, 1)
        params = update(params, grads)
        print(params)    
        from time import sleep
        sleep(0.1)


