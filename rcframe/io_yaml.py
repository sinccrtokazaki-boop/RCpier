"""YAML 入力ファイルの読み込み（書式は README.md 参照）"""
from pathlib import Path

import yaml

from .building import Building, InputError


def building_from_yaml(text: str) -> Building:
    """YAML 文字列から Building を生成する"""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise InputError("YAML のトップレベルは辞書形式で記述してください")
    return Building(data)


def load_building(path) -> Building:
    """YAML ファイルから Building を生成する"""
    return building_from_yaml(Path(path).read_text(encoding="utf-8"))
