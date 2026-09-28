"""YAML 入力ファイルの読み込み（書式は README.md 参照）"""
from pathlib import Path

import yaml

from .building import Building, InputError


def _load(text):
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise InputError("YAML のトップレベルは辞書形式で記述してください")
    return data


def building_from_yaml(text: str) -> Building:
    """YAML 文字列から Building を生成する"""
    bld = Building(_load(text))
    bld.source_text = text      # 計算書の付録に載せる
    return bld


def load_building(path) -> Building:
    """YAML ファイルから Building を生成する"""
    return building_from_yaml(Path(path).read_text(encoding="utf-8"))


def model_from_yaml(text: str):
    """YAML 文字列から、type に応じて Pier（type: pier）または Building を生成する"""
    data = _load(text)
    if str(data.get("type", "building")) == "pier":
        from .pier import Pier
        obj = Pier(data)
    else:
        obj = Building(data)
    obj.source_text = text
    return obj


def load_model(path):
    """YAML ファイルから Pier または Building を生成する"""
    return model_from_yaml(Path(path).read_text(encoding="utf-8"))
