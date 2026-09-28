"""rcframe: RC造 3次元フレーム解析システム"""
from .materials import Concrete, RectSection, SteelGrade, STEELS, Bars, Hoop, BeamRebar, ColumnRebar
from .model import Model, Node, Element
from .results import Result
from .building import Building, InputError
from .io_yaml import load_building, building_from_yaml
from .check import run_checks, CheckReport, CheckRow
from .pushover import run_pushover, pushover, PushoverResult
from .capacity import run_ultimate, UltimateReport


def write_report(bld, path):
    """計算書（HTML）を書き出す（report.py。python -m rcframe.report と両立させるため遅延 import）"""
    from .report import write_report as _write
    return _write(bld, path)


def build_report(bld):
    """計算書の HTML 文字列を返す"""
    from .report import build_report as _build
    return _build(bld)

__version__ = "0.1.0"
