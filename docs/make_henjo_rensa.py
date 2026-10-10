# -*- coding: utf-8 -*-
"""RC 桟橋（直杭式横桟橋）の変状連鎖図をつくる.

  python docs/make_henjo_rensa.py [ページ用 HTML の出力先]

出力：このフォルダに 変状連鎖図_RC桟橋.svg（調書・スライドに貼る用, 色は固定）と .png（Edge で描画, 2 倍解像度）。
引数を付けると、クリックで経路をたどれる公開ページ用の HTML（本文だけ）も書く。
ノード・矢印・説明は rcframe/pier_chain.py の表を直す。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rcframe import pier_chain  # noqa: E402
from rcframe.chain_draw import write_file_outputs, write_page  # noqa: E402

if __name__ == "__main__":
    pier_chain.topo_order()                       # 循環がないことの確認
    p_svg, p_png = write_file_outputs(pier_chain, HERE)
    print("SVG →", p_svg)
    print("PNG →", p_png)
    if len(sys.argv) > 1:
        print("ページ →", write_page(pier_chain, sys.argv[1]))
