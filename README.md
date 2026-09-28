# rcframe — RC造 3次元フレーム解析システム

実務の設計補助を目的とした、Python 製の RC 造立体骨組解析プログラムです。
現在はフェーズ5（非線形解析: 材端ばねモデルによるプッシュオーバー）まで実装しています。

## 動作環境
Python 3.10 以上、numpy、scipy、PyYAML（テストには pytest）

```
pip install -r requirements.txt
python -m pytest tests          # 理論解との照合テスト
python examples/two_story_frame.py
python examples/from_yaml.py examples/two_story_frame.yaml
python examples/from_yaml.py examples/two_story_auto_loads.yaml   # 荷重の自動生成
python examples/from_yaml.py examples/two_story_design.yaml       # 断面算定・プッシュオーバー
```

## 実装済みの機能
- 直接剛性法による3次元骨組解析（1節点6自由度、疎行列ソルバー）
- せん断変形を考慮した梁柱要素（Timoshenko 梁）
- コンクリートのヤング係数（RC規準式）、矩形断面の断面性能・ねじり定数
- 剛性増大率（スラブ協力幅の考慮）
- 剛域（任意指定、または RC規準の慣用値で自動設定）
- 剛床仮定（マスター・スレーブ拘束）
- 節点荷重、部材分布荷重（等分布・台形・三角形などの区分線形分布。剛域上の荷重、
  非対称荷重でのせん断変形の影響も厳密に考慮）
- 荷重ケースの線形重ね合わせ
- 断面力（フェイス位置／節点位置）、支点反力、層間変形角
- 通り芯・階・断面リストからのモデル自動生成（YAML 入力、剛域・剛床も自動）
- 荷重の自動生成: 床荷重の亀甲分割による梁への分配、部材自重、地震用重量と重心、
  Ai 分布による地震力（重心に作用）、偶発偏心
- 部材の任意位置の断面力（`Result.section_forces_at`）
- 断面算定（RC規準 2018）: 長期・短期の荷重組合せ、梁の曲げ・せん断、柱の軸力と曲げ・せん断、構造規定の警告
- 静的非線形解析（プッシュオーバー）: 材端ばねモデル（トリリニア骨格・最大点指向型）、
  事象追跡型の荷重増分法、長期荷重 → Ai 分布の水平力、ひび割れ・降伏の発生順序

## 入力ファイル（YAML）
例: [examples/two_story_frame.yaml](examples/two_story_frame.yaml)

```python
from rcframe import load_building
bld = load_building("examples/two_story_frame.yaml")
m = bld.model                                   # Model
r = m.analyze("Ex")
r.story_drifts(bld.drift_levels(), 0)           # 層間変形角
bld.column("X1", "Y1", "1F")                    # 通り芯名 → 要素番号
bld.beam("Y1", "X1", "X2", "2F"); bld.node("X1", "Y1", "2F"); bld.master("2F")
```

| 項目 | 書式 | 内容 |
|---|---|---|
| `name` | 文字列 | モデル名 |
| `materials` | `{名前: {Fc, gamma, nu, E}}` | コンクリート（Fc 必須） |
| `sections` | `{名前: {b, D, phi_y, phi_z, kappa, material}}` | 矩形断面（b, D 必須） |
| `defaults` | `{material: 名前}` | 材料の既定値 |
| `grids` | `{x: {通り名: 位置}, y: {…}}` | 通り芯（昇順。名前に `/ - *` は不可） |
| `floors` | `{階名: 床レベル z}` | 下から昇順。最下階の節点は固定支点 |
| `columns` / `beams` | `[{floor, at, section, material}]` | 配置ルール（下記） |
| `openings` | `[{floor, area}]` | 吹抜け（床のない区画） |
| `loads` | `{ケース: {beams: [{floor, at, w}], floors: {階名: [Fx, Fy, Mz]}}}` | 梁等分布荷重（全体座標 N/mm）、床の図心への集中荷重 |

**配置ルール** は上から順に適用し、後のルールが優先されます。
- `floor` : 階名またはそのリスト。省略すると全階。柱 `1F` = 1FL〜2FL 間の柱、梁 `2F` = 2FL 上の梁
- `at` : 位置（省略時 `"*"`）。リストも可
  - 柱 `"X2/Y1"`（交点）, `"X2/*"`, `"*/Y1"`, `"*"`
  - 梁 `"Y1/X1-X3"`（Y1通り上 X1〜X3 間の全スパン）, `"X2/*"`, `"*"`
  - 吹抜け `area: "X1-X2/Y1-Y2"`（X範囲/Y範囲）, `"*"`
- `section: none`（または null）で、その位置の部材を「なし」にする
- 材料は ルールの `material` → 断面の `material` → `defaults.material` の順

**自動設定**
- 節点は部材端となる交点にのみ作成
- 剛域は RC規準の慣用値（`Model.auto_rigid_zones`）
- 剛床: 最下階以外の各階で、床区画（通り芯で囲まれた区画 − 吹抜け）の面積図心にマスター節点を置き、
  床区画の角にある節点を従属させる。床区画に接しない節点は剛床に含めない

YAML の注意: `300e3` のような指数表記は PyYAML では文字列になりますが、数値欄はすべて数値に変換して読み込みます。

### 荷重の自動生成
例: [examples/two_story_auto_loads.yaml](examples/two_story_auto_loads.yaml)

**単位の例外**: 以下の荷重欄だけは実務の荷重表の値をそのまま書けるよう、
**kN/m²・kN/m³・kN** で入力します（読み込み時に N, mm へ換算。内部計算はすべて N, mm）。

| 項目 | 書式 | 内容 |
|---|---|---|
| `floor_loads` | `{用途: {DL, LL_frame, LL_seismic}}` [kN/m²] | 固定荷重（スラブ自重・仕上げ込み、必須）、積載荷重（架構用・地震用、省略時 0） |
| `floor_use` | `[{floor, area, use}]` | 床区画の用途。配置ルールと同様に後の行が優先、`use: none` で荷重なし |
| `self_weight` | `{gamma}` [kN/m³] | 梁・柱の自重（省略時 24） |
| `extra_weights` | `{階名: 重量}` [kN] | 地震用重量への加算（パラペット・外壁など）。床の図心に置く |
| `seismic` | `{Z, soil, C0, T, directions, eccentricity}` | Ai 分布の地震力（下記） |

`floor_loads`, `floor_use`, `self_weight`, `extra_weights`, `seismic` のいずれかがあると、
次の荷重ケースを自動生成します。手入力の `loads` に同じ名前のケースがあれば加算されます。

| ケース | 内容 |
|---|---|
| `G` | 固定荷重: 床の DL を梁へ分配 ＋ 梁・柱の自重（γ·b·D） |
| `P` | 積載荷重（架構用 LL_frame）を梁へ分配 |
| `Ex`, `Ey` | Ai 分布の地震力を各階の重心に作用 |
| `Ex+e`, `Ex-e`, `Ey+e`, `Ey-e` | 偶発偏心（`eccentricity` > 0 のとき） |

- **床荷重の分配**: 4辺に梁がある区画は亀甲分割（45°線、長辺に台形・短辺に三角形）、
  対向する 2辺だけに梁がある区画は一方向スラブ（各辺に等分布）。それ以外はエラー。吹抜けの区画は荷重なし
- **地震用重量 Wi**: 床の (DL + LL_seismic)×面積 ＋ その階の梁 ＋ 上下の柱の半分 ＋ extra_weights。
  同時に重心を求める（`bld.weights`）
- **地震力**（建築基準法施行令第88条、昭和55年建設省告示第1793号）:
  T = 0.02h（h: 建物高さ [m]、`T` で直接指定も可）、Rt（地盤種別 `soil` 1/2/3 → Tc = 0.4/0.6/0.8 s）、
  Ai = 1 + (1/√αi − αi)·2T/(1+3T)、Ci = Z·Rt·Ai·C0、Qi = Ci·ΣWi、床の外力 Pi = Qi − Qi+1
- **作用位置**: 各階の重心。マスター節点（床の図心）には Fx, Fy と重心のずれによる Mz として載荷
- **偶発偏心**: 重心を荷重と直交方向に ±e×（床の外形寸法）ずらしたケースを追加
- 計算表は `bld.print_seismic_table()` で表示（Wi, ΣWi, αi, Ai, Ci, Qi, Pi, 重心）

### 断面算定（配筋と検定）
例: [examples/two_story_design.yaml](examples/two_story_design.yaml)

```python
from rcframe import load_building, run_checks
bld = load_building("examples/two_story_design.yaml")
rep = run_checks(bld)          # 必要な荷重ケースはここで解析する
rep.print_summary()            # 部材ごとの曲げ・せん断の最大検定比
rep.print_detail()             # 位置・方向ごと
rep.ng, rep.warnings, rep.max_ratio
```

**配筋**は断面に `rebar` として書きます（配筋のない断面は検定しません）。
`4-D25` = 本数-呼び径（D10〜D41）、`2-D10@200` = 本数-呼び径@ピッチ。

```yaml
sections:
  G1:                           # 梁: 両端共通の end と中央の center
    b: 400
    D: 700
    rebar:
      main: SD345               # 主筋の鋼種
      shear: SD295              # せん断補強筋の鋼種
      dt: 70                    # 主筋重心から材縁まで [mm]
      end:    {top: 4-D25, bottom: 3-D25, stirrup: 2-D13@150}
      center: {top: 2-D25, bottom: 3-D25, stirrup: 2-D13@150}
  C1:                           # 柱: 柱頭・柱脚共通
    b: 600
    D: 600
    rebar:
      main: SD345
      shear: SD295
      dt: 70
      x: 5-D25                  # X方向の曲げに対する 1面の本数（隅筋を含む）
      y: 5-D25
      hoop_x: 2-D13@100         # X方向のせん断に効く帯筋
      hoop_y: 2-D13@100
```

階ごとに配筋を変えるときは断面名を分けて（`G1-2F` 等）配置ルールで割り当てます。

| 項目 | 書式 | 内容 |
|---|---|---|
| `steels` | `{鋼種: {F, ftL, ftL_D29}}` | 既定は SD295 / SD345 / SD390。既定表にない鋼種は F 必須 |
| `design` | `{shear_factor, young_ratio, long, seismic}` | 短期せん断の割増し n（既定 1.5）、ヤング係数比（既定 15）、長期・地震の荷重ケース名（既定 G, P と E で始まるケース） |

**検定の内容**（日本建築学会「鉄筋コンクリート構造計算規準・同解説」2018年版）
- 許容応力度（6条）: コンクリート fc = Fc/3（短期 2Fc/3）、fs = min(Fc/30, 0.49+Fc/100)（短期 1.5 倍）。
  主筋 長期 SD295 195、SD345・SD390 215（D29 以上 195）、短期 F。せん断補強筋 長期 195、短期 F（上限 390）
- 荷重組合せ（施行令第82条）: 長期 G+P、短期 G+P±E（地震の各ケース・偶発偏心ケース）
- 梁（両端フェイス・中央）: 曲げは複筋梁の許容曲げモーメント（13条、軸力無視）、
  せん断は QAL = b·j·α·fs、QAS = b·j·{(2/3)α·fs + 0.5·wft·(pw−0.002)}
- 柱（柱脚・柱頭フェイス、X・Y 方向を独立に）: 軸力 N を考慮した許容曲げモーメント Ma(N)（14条）、
  せん断は QAL = b·j·α·fs、QAS = b·j·{fs + 0.5·wft·(pw−0.002)}
- 短期の設計用せん断力 QD = QL + n·QE（15条）、α = 4/(M/(Q·d)+1)（1〜2）、j = 7d/8、pw ≦ 1.2%
- 許容曲げモーメントは平面保持・弾性（n = 15、コンクリートの引張無視）で、コンクリート圧縮縁・
  引張鉄筋・圧縮鉄筋のいずれかが許容応力度に達するときの値
- 構造規定の警告: 梁・柱の pw ≧ 0.2%、柱の pg ≧ 0.8%

### 静的非線形解析（プッシュオーバー）
例: [examples/two_story_design.yaml](examples/two_story_design.yaml)（`nonlinear` の項）

```python
from rcframe import load_building, run_pushover
bld = load_building("examples/two_story_design.yaml")
po = run_pushover(bld, "+x")     # "+x", "-x", "+y", "-y"
po.print_curve()                 # 荷重係数・ベースシア・各層の層間変形角・事象
po.print_hinges()                # ひび割れ・降伏した材端ばねと発生ステップ（all_springs=True で全て）
po.steps, po.events, po.final    # final: 最終状態の Result（section_forces 等が使える）
```

```yaml
nonlinear:
  sigma_y_factor: 1.1         # 鉄筋の材料強度 σy = 1.1F
  post_yield_ratio: 0.001     # 降伏後の剛性比（部材の初期剛性に対する）
  pushover:
    directions: [+x, -x, +y, -y]
    max_drift: 0.02           # いずれかの層の層間変形角がこの値に達したら終了
    step_drift: 0.0002        # 1ステップの層間変形角増分の上限（曲線の細かさ）
```

- **部材モデル**: 弾性部材（Timoshenko）の可撓部両端に曲げの回転ばね。梁は鉛直面内の曲げのみ、
  柱は X・Y 方向それぞれ独立。軸・せん断・ねじりは弾性。配筋のない部材は弾性
- **骨格曲線**（2020年版 建築物の構造関係技術基準解説書 付録1-3 の略算式）:
  Mc = 0.56√Fc·Ze（柱は + N·D/6）、梁 My = 0.9·at·σy·d、柱 My は軸力による 3区分の式、
  降伏点剛性低下率 αy は菅野式（せん断スパン = 内法長さ/2、pt = at/(b·d)）。
  部材の逆対称曲げの M–θ（初期剛性 K0、降伏点 My/(αy·K0)、降伏後 r·K0）が
  トリリニアとなるようにばねの骨格を定める。柱の軸力は長期荷重時の値で固定
- **履歴則**: 最大点指向型。除荷時はばね剛、M が 0 を越えたら反対側の最大点（未経験ならひび割れ点）へ向かう
- **解析法**: 事象追跡型の荷重増分法。各ステップは次のばねの折れ点まで（ばねが折れ線なので厳密で、
  不釣合い力や収束計算がない）。長期荷重 G+P を載荷後、Ai 分布の水平力（重心のずれの Mz を含む）を比例増加

部材荷重を Python から与える場合は `Model.add_member_load(case, eid, w, profile=[(x, 係数), ...])`
で区分線形の分布を指定できます（x は i節点からの距離 [mm]）。

## 規約
- 単位: mm, N（表示は kN, kN·m）。例外として YAML の荷重欄（floor_loads 等）は kN/m²・kN/m³・kN
- 全体座標: Z 軸が鉛直上向き
- 部材局所座標: x = i端→j端
  - 水平梁: 局所 z = 鉛直上向き（せい D の方向）
  - 柱（i端を下にする）: 局所 y = 全体 X、局所 z = 全体 Y
  - beta（コードアングル）で局所 x 軸まわりに回転可
- 断面力: N は引張正。i端 = −(材端力)、j端 = +(材端力)

## 開発ロードマップ
| フェーズ | 内容 | 状態 |
|---|---|---|
| 1 | 線形弾性解析エンジン（剛域・剛床・せん断変形） | 済 |
| 2 | 入力の効率化（通り芯・階からのモデル自動生成、入力ファイル形式） | 済 |
| 3 | 荷重の自動生成（床荷重の梁への分配、Ai分布による地震力、偏心） | 済 |
| 4 | 断面算定（RC規準: 曲げ・せん断の許容応力度検定） | 済 |
| 5 | 非線形解析（材端剛塑性ばね、トリリニア骨格曲線、荷重増分法） | 済 |
| 6 | 保有水平耐力（Ds・Fes、崩壊形の判定） | 予定 |
| 7 | 3D可視化・計算書出力 | 予定 |

## 既知の簡略化
- 柱の自動剛域は、3次元で方向ごとに異なる値の最小値を採用
- 柱梁接合部の変形、壁・ブレースは未対応
- 剛床は 1 つの階に 1 つ（吹抜けで床が分断されても 1 つの剛床として扱う）
- 自動生成は直交する通り芯のみ（斜め通り芯、片持ち梁、小梁、階ごとのレベル差は未対応）
- 床荷重は通り芯で囲まれた区画単位（小梁による区画の分割、片持ちスラブは未対応）
- 梁の自重は b×D で算定し、スラブとの重複を差し引かない（安全側）
- 地震力は RC造として T = 0.02h（α = 0）。剛床のない階がある建物には Ai 分布の地震力を作用させられない
- 断面算定: 柱の 2軸曲げの同時考慮、付着、柱梁接合部、曲げ降伏時せん断力による設計は未対応。
  梁の軸力・弱軸曲げは無視。柱の曲げは各方向の 1面の主筋のみ考慮（中段筋は無視）。
  梁の中央はスパン中央の値で検定（スパン内の最大正曲げ位置ではない）。梁の左右端部は同一配筋
- 非線形解析: 曲げばねのみ（せん断破壊・せん断ばねは未考慮）、柱の変動軸力による My の変化と
  2方向曲げの相互作用は未考慮、P-Δ 効果なし、梁の My はスラブ筋の寄与を無視、Ze は鉄筋を無視
