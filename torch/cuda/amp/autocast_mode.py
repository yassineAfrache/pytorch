import functools
import warnings
from typing import Any

import torch

__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        warnings.warn(
            "`torch.cuda.amp.autocast(args...)` is deprecated. "
            "Please use `torch.amp.autocast('cuda', args...)` instead.",
            DeprecationWarning,
        )
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    warnings.warn(
        "torch.cuda.amp.custom_fwd(args...) is deprecated. Please use torch.amp.custom_fwd(args..., device_type='cuda') instead."
    )
    return functools.partial(torch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


def custom_bwd(bwd):
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    warnings.warn(
        "torch.cuda.amp.custom_bwd(args...) is deprecated. Please use torch.amp.custom_bwd(args..., device_type='cuda') instead."
    )
    return functools.partial(torch.amp.custom_bwd, device_type="cuda")(bwd)
