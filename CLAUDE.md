# CLAUDE.md — rcframe プロジェクトの指針

## 概要
RC造の3次元フレーム解析システム（Python）。実務の設計補助ツールとして、
最終的に非線形静的解析（プッシュオーバー）と保有水平耐力の確認までを目指す。
回答・コメント・docstring は日本語で書くこと。

## コマンド
- テスト: `python -m pytest tests -q`
- 計算例: `python examples/two_story_frame.py` / `python examples/from_yaml.py [入力.yaml]`
- 計算書: `python -m rcframe.report 入力.yaml -o report.html`
- 桟橋: `python examples/pier_example.py [入力.yaml]`

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
- `rcframe/skeleton.py`  : 材端ばねの骨格曲線（Mc, My, 菅野式 αy、技術基準解説書 付録1-3）
- `rcframe/hinge.py`     : ばねの履歴則（最大点指向型、除荷剛）。接線柔性と次の事象を返す
- `rcframe/pushover.py`  : 事象追跡型の荷重増分解析（本体 pushover は入力形式に非依存、run_pushover は Building 用）
  - ばねは Element.springs（柔性 [面][端]）で element.py の local_stiffness_springs に渡す。解析後は None に戻す
- `rcframe/ultimate.py`  : 荒川式 Qsu、部材種別・部材群・Ds、剛性率・偏心率・Fs・Fe（計算のみ）
- `rcframe/capacity.py`  : 保有水平耐力の確認（run_ultimate → UltimateReport）、剛性率・偏心率（stiffness_eccentricity）
- `rcframe/svg.py`       : 計算書の図（軸組図・伏図・モーメント図・曲線）。モーメント図は引張側に描く
- `rcframe/grid.py`      : 通り芯と位置指定の解釈（GridMixin）、入力チェックの小道具（建物・桟橋で共通）
- `rcframe/steel.py`     : 桟橋用の材料・断面（Steel、PipeSection、JsceConcrete）、Chang の β
- `rcframe/pier.py`      : 桟橋（直杭式横桟橋）のモデル生成（Pier）。load_model は type: pier で Pier を返す
  - 荷重の生成（actions: D, L, Ls, B, T, EQ）と組合せ（combinations、部分係数は入力）
- `rcframe/port_loads.py`: 桟橋の荷重の計算（てこの原理の配分、偏心モーメント、牽引力の成分、重量の集計）
- `rcframe/port_check.py`: 桟橋の照査（示方書 2022 の Mud・Vyd・Vwcd・ひび割れ幅、鋼管杭、支持力）。
  照合が必要な事項は VERIFY_ITEMS
- 配筋の読み込みは building.parse_rebar（建物・桟橋で共通）
- `rcframe/report.py`    : 計算書 HTML（1ファイル完結、外部参照禁止）。計算は既存関数を呼ぶだけで整形のみ
  - 規準書で照合が必要な事項は VERIFY_ITEMS、既知の簡略化は KNOWN_SIMPLIFICATIONS（README と揃える）

## 規約（変更しないこと）
- 単位: mm, N。表示のみ kN, kN·m
  - 例外: YAML の荷重欄（floor_loads, self_weight, extra_weights）は kN/m²・kN/m³・kN で入力し、
    読み込み時に換算する（building.py の KN_M2 等）。内部計算は常に mm, N
  - 桟橋の YAML: 高さ・通り芯 m、断面寸法 mm、荷重 kN・kN/m・kN·m、kh kN/m³（pier.py の M, KN 等で換算）
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
→ 4 済: 断面算定（RC規準 2018）→ 5 済: 非線形（材端ばね・増分法）→ 6 済: 保有水平耐力 → 7 済: 可視化・計算書（HTML）

## 桟橋編のロードマップ（README.md 参照）
基準: 港湾基準（2018）、コンクリート標準示方書（2022）。部分係数はプログラムに組み込まず YAML で入力する
P1 済: モデル生成（仮想固定点）→ P2 済: 荷重・組合せ → P3 済: 照査 → P4: レベル2（プッシュオーバー）→ P5: 計算書

## 既知の簡略化
- 柱の自動剛域は方向ごとの値の最小値を採用
- 柱梁接合部変形、壁・ブレースは未対応
- 剛床は 1 階につき 1 つ（床が分断されても 1 つ）。自動生成は直交通り芯のみ
- 床荷重は通り芯区画単位（小梁・片持ちスラブ未対応）。梁自重は b×D（スラブ重複を控除しない）
- 断面算定: 柱は X・Y 独立（2軸曲げ未対応）、中段筋無視、付着・接合部・曲げ降伏時せん断は未対応
- 非線形: 曲げばねのみ、柱軸力は長期で固定、P-Δ なし
- 保有水平耐力: 耐力壁なし（純ラーメンの Ds のみ）、FD の柱があれば部材群 D（安全側）、局部崩壊は判定しない
- 桟橋: 杭は仮想固定点で固定（地盤ばねなし）、上部工は 1つの剛床、梁端の剛域なし
- 桟橋の荷重: 張出しは端の梁への線荷重（ねじり無視）、防舷材・係船柱はてこの原理で節点へ、杭の慣性力なし
- 照合が必要（桟橋）: 示方書の Ec の表（steel.JSCE_EC_TABLE）、鋼管のせん断形状係数 2
