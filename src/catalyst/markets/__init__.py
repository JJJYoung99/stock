from catalyst.markets.base import MarketAdapter
from catalyst.markets.sample import SampleAdapter

__all__ = ["MarketAdapter", "SampleAdapter", "get_adapter"]


def get_adapter(kind: str, **kwargs) -> MarketAdapter:
    """Factory. `kind` is one of: 'sample', 'kr', 'us'.

    The KR / US adapters require optional dependencies and are imported
    lazily so the core package stays installable without them.
    """
    m = kind.lower()
    if m == "sample":
        return SampleAdapter(**kwargs)
    if m == "kr":
        from catalyst.markets.kr import KoreaAdapter
        return KoreaAdapter(**kwargs)
    if m == "us":
        from catalyst.markets.us import USAdapter
        return USAdapter(**kwargs)
    raise ValueError(f"unknown market: {market!r}")
