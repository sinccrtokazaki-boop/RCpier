# CLAUDE.md — rcframe プロジェクトの指針

## 概要
RC造の3次元フレーム解析システム（Python）。実務の設計補助ツールとして、
最終的に非線形静的解析（プッシュオーバー）と保有水平耐力の確認までを目指す。
回答・コメント・docstring は日本語で書くこと。

## コマンド
- テスト: `python -m pytest tests -q`
- 計算例: `python examples/two_story_frame.py` / `python examples/from_yaml.py [入力.yaml]`

## 構成
- `rcframe/materials.py` : Concrete（RC規準式のE）, RectSection（剛性増大率 phi_y/phi_z）, 鋼種・配筋
- `rcframe/element.py`   : 12x12 局所剛性（Timoshenko）、座標変換、剛域変換、等分布・区分線形分布荷重
- `rcframe/model.py`     : Model（入力、剛域自動設定、剛床拘束、組立、求解）
- `rcframe/results.py`   : Result（断面力、任意位置の断面力、反力、層間変形角、荷重組合せ）
- `rcframe/building.py`  : Building（通り芯・階・配置ルールから Model を生成、剛床・荷重も設定）
- `rcframe/io_yaml.py`   : YAML → dict → Building（書式は README.md）
- `rcframe/loads.py`     : 亀甲分割の負担幅、T・Rt・Ai・地震層せん断力（入力形式に依存しない計算のみ）
- `rcframe/design.py`    : RC規準 2018 の許容応力度・許容曲げモーメント Ma(N)・許容せん断力（計算のみ）
- `rcframe/check.py`     : 荷重組合せと全部材の検定（run_checks → CheckReport）

## 規約（変更しないこと）
- 単位: mm, N。表示のみ kN, kN·m
  - 例外: YAML の荷重欄（floor_loads, self_weight, extra_weights）は kN/m²・kN/m³・kN で入力し、
    読み込み時に換算する（building.py の KN_M2 等）。内部計算は常に mm, N
- 全体座標は Z 上向き。局所 x = i端→j端
- 水平梁: 局所 z = 鉛直上向き。柱（i端が下）: 局所 y = 全体X, 局所 z = 全体Y
- 局所自由度順: [ux, uy, uz, rx, ry, rz] × (i, j)
- 断面力: N 引張正、i端 = −材端力、j端 = +材端力
- 部材端力は "face"（剛域端）と "node"（節点位置）の両方を保持

## 開発ルール
- 新機能には必ず理論解または手計算と照合するテストを tests/ に追加する
- 変更後は全テストが通ることを確認してから完了とする
- 部材剛性の計算（element.py）と全体の組立・求解（model.py）は分離を保つ
  （非線形化の際に要素剛性だけを差し替えられるようにするため）
- 規準式を実装するときは、出典（規準名・式番号）をコメントに残す

## ロードマップ（README.md 参照）
1 済: 線形弾性エンジン → 2 済: モデル自動生成（YAML） → 3 済: 荷重自動生成（Ai分布等）
→ 4 済: 断面算定（RC規準 2018）→ 5: 非線形（材端ばね・増分法）→ 6: 保有水平耐力 → 7: 可視化・計算書

## 既知の簡略化
- 柱の自動剛域は方向ごとの値の最小値を採用
- 柱梁接合部変形、壁・ブレースは未対応
- 剛床は 1 階につき 1 つ（床が分断されても 1 つ）。自動生成は直交通り芯のみ
- 床荷重は通り芯区画単位（小梁・片持ちスラブ未対応）。梁自重は b×D（スラブ重複を控除しない）
- 断面算定: 柱は X・Y 独立（2軸曲げ未対応）、中段筋無視、付着・接合部・曲げ降伏時せん断は未対応
