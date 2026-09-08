"""SevenNet-owned e3nn convolution/gate region selection."""
from md_benchmark.opt4_fx import install_tp_regions


def install(model, passes, report):
    install_tp_regions(model, passes, report,
        lambda path: "convolution" in path or "self_connection" in path or "gate" in path)
