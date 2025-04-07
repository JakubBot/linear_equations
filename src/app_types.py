from typing import TypedDict

class Config(TypedDict):
    x_label: str
    y_label: str
    title: str
    path: str
    log_y_axis: bool
    has_x_axis: bool
    plot: any
    axhline: any
    figsize: tuple[float, float]