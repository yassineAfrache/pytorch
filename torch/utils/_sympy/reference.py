import math

import operator

import sympy

import torch
from torch.utils._sympy.functions import (
    _keep_float,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    ToFloat,
    TruncToInt,
)


# The sympy interpretation of operators.  It will also sometimes work with
# plain int/float, but if you do certain operations you will get out a
# sympy.Basic in the end.  If you want the Python/FX traceable interpretation,
# check PythonReferenceAnalysis.
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    @staticmethod
    def constant(c, dtype):
        return sympy.sympify(c)

    @staticmethod
    def or_(a, b):
        return a | b

    @staticmethod
    def and_(a, b):
        return a & b

    @staticmethod
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        return a < b

    @staticmethod
    def gt(a, b):
        return a > b

    @staticmethod
    def le(a, b):
        return a <= b

    @staticmethod
    def ge(a, b):
        return a >= b

    @staticmethod
    def not_(a):
        assert not isinstance(a, bool)
        return ~a

    @staticmethod
    def reciprocal(x):
        return FloatTrueDiv(1.0, x)

    @staticmethod
    def square(x):
        return x * x

    @staticmethod
    def to_int(x):
        return TruncToInt(x)

    @staticmethod
    def to_float(x):
        return ToFloat(x)

    @staticmethod
    def mod(x, y):
        return Mod(x, y)

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return FloatTrueDiv(a, b)

    @staticmethod
    def int_truediv(a, b):
        return IntTrueDiv(a, b)

    @staticmethod
    def floordiv(a, b):
        return FloorDiv(a, b)

    @staticmethod
    def truncdiv(a, b):
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    def add(a, b):
        return _keep_float(operator.add)(a, b)

    @staticmethod
    def mul(a, b):
        return _keep_float(operator.mul)(a, b)

    @staticmethod
    def sub(a, b):
        return _keep_float(operator.sub)(a, b)

    @staticmethod
    def exp(x):
        return OpaqueUnaryFn_exp(x)

    @staticmethod
    def log(x):
        return OpaqueUnaryFn_log(x)

    @staticmethod
    def sqrt(x):
        return OpaqueUnaryFn_sqrt(x)

    @staticmethod
    def pow(a, b):
        return _keep_float(FloatPow)(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        return PowByNatural(a, b)

    @staticmethod
    def minimum(a, b):
        return sympy.Min(a, b)

    @staticmethod
    def maximum(a, b):
        return sympy.Max(a, b)

    @staticmethod
    def floor(x):
        return sympy.floor(x)

    @staticmethod
    def ceil(x):
        return sympy.ceiling(x)


# Unlike ReferenceAnalysis, does NOT sympyify, instead, works with plain
# Python types and is FX traceable.  Inheritance here is purely for code
# sharing (TODO: considering splitting out a BaseReferenceAnalysis).
class PythonReferenceAnalysis(ReferenceAnalysis):
    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def to_float(a):
        return float(a)

    @staticmethod
    def exp(x):
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x):
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def sqrt(x):
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    @staticmethod
    def floor(x):
        return math.floor(x)

    @staticmethod
    def ceil(x):
        return math.ceil(x)

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def pow_by_natural(a, b):
        return a**b
