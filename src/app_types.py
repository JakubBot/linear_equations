from typing import TypedDict

class Config(TypedDict):
    x_label: str
    y_label: str
    title: str
    path: str
    plot: any
    log_y_axis: bool