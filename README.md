# rcframe — RC造 3次元フレーム解析システム

実務の設計補助を目的とした、Python 製の RC 造立体骨組解析プログラムです。
現在はフェーズ7（3D表示つき計算書の HTML 出力）まで、ロードマップのすべてを実装しています。

## 動作環境
Python 3.10 以上、numpy、scipy、PyYAML（テストには pytest）

```
pip install -r requirements.txt
python -m pytest tests          # 理論解との照合テスト
python examples/two_story_frame.py
python examples/from_yaml.py examples/two_story_frame.yaml
python examples/from_yaml.py examples/two_story_auto_loads.yaml   # 荷重の自動生成
python examples/from_yaml.py examples/two_story_design.yaml       # 断面算定・プッシュオーバー・保有水平耐力
python -m rcframe.report examples/two_story_design.yaml -o report.html   # 計算書（HTML）
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
- 保有水平耐力の確認: せん断終局強度（荒川式）と破壊形式、部材種別・部材群・Ds、
  剛性率・偏心率と Fes、崩壊機構の検出、Qu ≧ Qun の判定
- 計算書の HTML 出力: 1ファイルで完結（外部参照なし）、3D モデルの回転・変形図・ヒンジ表示、
  伏図・軸組図・曲げモーメント図・検定比図・ヒンジ図・荷重変形曲線、A4 印刷用 CSS

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

### 保有水平耐力（令82条の3）
例: [examples/two_story_design.yaml](examples/two_story_design.yaml)（`ultimate` の項）

```python
from rcframe import load_building, run_ultimate
bld = load_building("examples/two_story_design.yaml")
rep = run_ultimate(bld)          # ±X, ±Y のプッシュオーバーを実行して判定
rep.print_summary()              # 層・方向ごとの Qu, Qud, 部材群, Ds, Fs, Fe, Fes, Qun, Qu/Qun
rep.print_members()              # 部材の Qmu, Qsu, 破壊形式, 部材種別
rep.print_eccentricity()         # 剛性率・偏心率の計算表
rep.ok, rep.stories, rep.pushovers
```

```yaml
ultimate:
  qsu: min                    # 荒川 min 式 / mean 式
  qsu_factor: 1.0             # 曲げ破壊先行の条件 Qsu ≧ 係数·Qmu
  drift_limit: 0.01           # Qu を決める層間変形角の上限
  stop_at_column_shear: true  # 柱のせん断破壊の時点で Qu を決める
  ds_override: {1F: 0.35}     # Ds を手動で指定する層（任意）
```

- **Qu**: Ai 分布の水平力によるプッシュオーバーで、崩壊機構の形成・層間変形角の上限・柱のせん断破壊
  のうち最初の時点の各層の層せん断力（= 荷重係数 × Ai 分布の Qi）。
  崩壊機構は、降伏したばねを解放（ピン）した接線剛性が水平力に対して不安定になることで検出し、
  変形が集中する層から全体崩壊形・部分崩壊形を表示
- **Qud** = C0 = 1.0 の地震層せん断力（Ai 分布の Qi / C0）、**Qun** = Ds·Fes·Qud
- **せん断終局強度**（荒川式）: Qsu = {k·pt^0.23·(Fc+18)/(M/(Q·d)+0.12) + 0.85√(pw·σwy) + 0.1σ0}·b·j、
  k = 0.053（min）/ 0.068（mean）、M/(Q·d) = 内法長さ/(2d)（1〜3）、pw ≦ 1.2%、σwy = F、σ0 は長期軸力
- **破壊形式**: Qmu = 両端の My の和 / 内法長さ（梁は長期せん断力を加算）。Qsu < 係数·Qmu、
  または解析中にせん断力が Qsu に達した部材はせん断破壊（FD）
- **部材種別**（昭55建告第1792号 第4）: 柱は h0/D, σ0/Fc, pt, τu/Fc、梁は τu/Fc（τu = Qmu/(b·D)）。
  柱の種別は柱頭・柱脚に接続する加力方向の梁の種別のうち低い方も考慮
- **部材群**: Qu 時の柱のせん断力の割合 γ から A（γA ≧ 0.5 かつ γC ≦ 0.2）/ B（γC < 0.5）/ C（γC ≧ 0.5）、
  FD の柱があれば D。**Ds**（RC 純ラーメン）A 0.30, B 0.35, C 0.40, D 0.45
- **Fes = Fs·Fe**（令82条の6、昭55建告第1791号・第1792号 第7）: 一次設計の線形解析（Ex, Ey）の層間変形から
  剛性率 Rs、柱の水平剛性 D（柱のせん断力 / 層間変位）から剛心・ねじり剛性・弾力半径・偏心率 Re

### 計算書（HTML）
```
python -m rcframe.report input.yaml -o report.html
```
```python
from rcframe import load_building, write_report
write_report(load_building("input.yaml"), "report.html")
```

- CSS・JavaScript・図をすべて埋め込んだ 1つの HTML ファイル（インターネット接続なしでブラウザで開ける）
- 入力の内容に応じて章を出力: 建物概要（伏図・全体図）、荷重、応力解析（3D 表示、軸組図の曲げモーメント図、
  支点反力）、層間変形角・剛性率・偏心率、断面検定（検定比の表と軸組図）、プッシュオーバー（荷重変形曲線、
  ヒンジ図）、保有水平耐力、付録（照合が必要な事項・既知の簡略化・警告・入力データ）。
  配筋がなければ断面検定以降、ultimate / nonlinear の指定がなければ該当章を省略
- 3D 表示: ドラッグで回転、ホイールで拡大縮小、荷重ケース・プッシュオーバー最終状態の変形図（倍率調整）、
  ひび割れ・降伏したばねの表示
- 印刷（A4）: 章ごとに改ページ、3D 表示は印刷せず全体図（静止図）を用いる
- 表紙は任意: `report: {title, author, date, notes}`

部材荷重を Python から与える場合は `Model.add_member_load(case, eid, w, profile=[(x, 係数), ...])`
で区分線形の分布を指定できます（x は i節点からの距離 [mm]）。

## 桟橋編（直杭式横桟橋・鋼管杭）
港湾の桟橋の RC フレーム解析。建物編と同じ解析エンジンを使い、入力と荷重・照査を桟橋用にする。
基準: 港湾の施設の技術上の基準・同解説（2018年版）、コンクリート標準示方書（2022年版）。

| 段階 | 内容 | 状態 |
|---|---|---|
| P1 | 桟橋モデルの自動生成（上部工の梁、鋼管杭、仮想固定点）と線形解析 | 済 |
| P2 | 荷重（自重・上載荷重・接岸力・牽引力・レベル1地震動）と部分係数による組合せ | 済 |
| P3 | 照査（上部工 RC、鋼管杭、杭の支持力） | 済 |
| P4 | レベル2地震動（プッシュオーバー）、地盤ばねモデル | 済 |
| P5 | 計算書（HTML） | 済 |

```
python examples/pier_example.py examples/pier_example.yaml    # 仮想固定点モデル（P1〜P3）
python examples/pier_example.py examples/pier_level2.yaml     # 地盤ばねモデルとレベル2（P4）
python -m rcframe.report examples/pier_level2.yaml -o pier_report.html   # 計算書（P5）
```

**計算書（P5）**: 建物編と同じ 1ファイル完結の HTML。`python -m rcframe.report` と `write_report` は
入力の `type` で建物・桟橋を自動で切り替える。章: 概要（諸元・平面図・断面図・杭の諸元）、荷重と組合せ、
解析結果（3D 表示、組合せごとの変位・杭の軸力と曲げ、Y方向の断面の曲げモーメント図）、照査、
レベル2（荷重–変位曲線・判定・全塑性の発生順序・塑性ヒンジ図）、付録（照合が必要な事項・既知の簡略化・警告・入力データ）。
照査・レベル2の指定がなければその章は省略。レベル2を含む計算例の出力には 1分程度かかる
```python
from rcframe import load_model
pier = load_model("examples/pier_example.yaml")   # type: pier なら Pier、それ以外は Building
r = pier.model.analyze("H")
pier.pile_info("B1", "L")      # β, 仮想固定点, 杭長, 自由長
pier.pile("B1", "L"); pier.beam("L", "B1", "B2"); pier.node("B1", "L"); pier.master()
```

例: [examples/pier_example.yaml](examples/pier_example.yaml)。**単位は建物編と異なる**:
高さ・通り芯 m（D.L. 基準）、断面寸法 mm、荷重 kN・kN/m・kN·m、kh kN/m³（内部は mm, N）。

| 項目 | 書式 | 内容 |
|---|---|---|
| `type` | `pier` | 桟橋の入力であることを示す |
| `materials` | `{名前: {fck, E, nu, gamma}}` / `{名前: {steel: true, fy, E, nu}}` | 上部工コンクリート（Ec は示方書の表から補間）/ 鋼材 |
| `sections` | `{名前: {b, D, material}}` / `{名前: {pipe: {D, t}, material, corrosion: {outer, inner}, stiffness}}` | 上部工の梁 / 鋼管杭（腐食代 mm、解析の剛性 intact / corroded） |
| `grids` | `{x: {名前: 位置 m}, y: {…}}` | x = 法線方向の杭位置、y = 陸側 → 海側の杭の列 |
| `deck` | `{top, model_level, soffit}` | 上部工の天端・骨組の高さ（梁の中心）・下面 [m] |
| `beams` | `[{at, section}]` | 縦梁・横梁の配置ルール（建物編と同じ位置指定） |
| `piles` | `{section, rows: {列: {seabed, kh}}, overrides: [{at, section}]}` | 杭の断面、列ごとの海底面 [m] と kh [kN/m³]、個別の上書き（none で杭なし） |
| `loads` | `{ケース: {deck: [Fx, Fy, Mz], beams: [{at, w}], nodes: [{at, F}]}}` | 手入力の荷重（上部工の図心・梁の等分布・節点）。actions と同名なら加算 |
| `deck.slab_thickness`, `deck.overhang` | m、`{sea, land}` m | 床版厚（自重用）、通り芯の外側への張出し |
| `unit_weight` | `{rc}` kN/m³ | 鉄筋コンクリートの単位体積重量（既定 24.5） |
| `actions` | 下記 | 荷重の自動生成 |
| `combinations` | `{名前: {state: 永続/変動/偶発, factors: {ケース: 係数}}}` | 部分係数による組合せ（係数は入力。逆方向は負の係数） |

**荷重の自動生成（actions）**: ケース名は固定。座標は X = 法線方向、Y = 陸側 → 海側（海側が +Y）

| ケース | 入力 | 生成方法 |
|---|---|---|
| `D` | `{pavement}` kN/m² | 床版（γ·t）と舗装を面荷重として梁へ（区画は亀甲分割、張出しは端の梁へ線荷重）、梁は床版より下の部分 γ·b·(D − t) |
| `L`, `Ls` | `{surcharge}` kN/m² | 上載荷重（常時・地震時）を D と同様に分配 |
| `B` | `{fenders: [{x, z, R, friction}]}` | 海側前面・高さ z の防舷材に反力 R（−Y）と摩擦 μR（+X、負の μ で −X）。海側の梁の両端節点へてこの原理で配分し、偏心のモーメント r × F を加える |
| `T` | `{bollards: [{x, P, angle_h, angle_v}]}` | 海側前面・天端の係船柱に牽引力（+Y 向き、法線直角から +X 側への水平角、仰角）。B と同様に節点へ |
| `EQx`, `EQy`（`EQz`） | `EQ: {kh, kv, directions}` | レベル1地震動（震度法）。kh ×（D + Ls）を上部工の重心に作用（剛床の図心とのずれの Mz）。kv > 0 なら EQz = kv ×（D + Ls の荷重） |

```python
res = pier.analyze_cases()                 # {ケース: Result}
comb = pier.analyze_combinations(res)      # {組合せ名: Result}
pier.weights, pier.seismic                 # 重量と重心、地震用重量
```

**照査（P3）**: `run_pier_checks(pier)` → `rep.print_summary()`, `rep.ng`, `rep.max_ratio`

```yaml
sections:
  G1: {b: 1200, D: 1500, material: C24, cover: 70,       # かぶり（主鉄筋の表面まで）[mm]
       rebar: {main: SD345, shear: SD345, dt: 100,
               end: {top: 10-D32, bottom: 6-D29, stirrup: 4-D16@200},
               center: {top: 6-D29, bottom: 10-D32, stirrup: 4-D16@250}}}
piles:
  rows: {L: {seabed: -4.0, kh: 20000, Rcd: 5000, Rtd: 2000}}   # 設計支持力（押込み・引抜き）[kN]
checks:                     # 係数は見本（既定値も同じ）。示方書・港湾基準で要確認
  gamma_i: 1.0
  concrete: {gamma_c: 1.3, gamma_s: 1.0, gamma_b_flexure: 1.1, gamma_b_vc: 1.3, gamma_b_vs: 1.1, gamma_b_vw: 1.3}
  crack: {combinations: [永続_常時], wa_ratio: 0.0035, eps_csd: 150.0e-6}
  pile: {gamma_m: 1.0}
  strength_combinations: all          # 断面破壊の照査（all = 永続・変動の全組合せ）
```

| 対象 | 照査 | 式 |
|---|---|---|
| 上部工の梁（杭の前面・中央） | 断面破壊（曲げ）γi·Md/Mud | 等価応力ブロック（k1 = 0.85, β = 0.8, ε'cu = 0.0035）とひずみの適合条件、Mud = Mu/γb |
| | 断面破壊（せん断）γi·Vd/Vyd、γi·Vd/Vwcd | Vcd = βd·βp·fvcd·bw·d/γb、Vsd = Aw·fwyd·(d/1.15)/(s·γb)、Vwcd = 1.25√f'cd·bw·d/γb |
| | 使用性 w/wa | w = 1.1·k1·k2·k3·{4c + 0.7(cs − φ)}·(σse/Es + ε'csd)、σse はひび割れ断面の弾性計算、wa = 係数 × c |
| 鋼管杭（杭頭・仮想固定点、腐食後の断面） | 軸力と曲げ | γi·(N/(A·fyd) + M/(Z·fyd))、M は 2方向の合成 |
| | せん断 | γi·(2Q/A)/(fyd/√3) |
| 杭の支持力 | 押込み・引抜き | γi·N/Rcd、γi·N/Rtd（Rcd, Rtd は入力値） |

組合せの荷重係数は combinations で考慮済みとし、照査では構造物係数 γi を乗じる（構造解析係数 γa = 1.0）。
式の形・係数のうち照合が必要な事項は `rcframe/port_check.py` の `VERIFY_ITEMS` に列挙。

**地盤ばねモデルとレベル2（P4）**: 例 [examples/pier_level2.yaml](examples/pier_level2.yaml)

```yaml
piles:
  model: soil_spring              # fixed_point（仮想固定点、既定）/ soil_spring（地盤ばね）
  segment: 0.5                    # 地中部の分割長 [m]
  rows:
    L: {seabed: -4.0, kh: 20000, tip: -24.0, pu: 300}   # tip: 杭先端（省略時 海底面 − 5/β）、pu: 地盤反力度の上限 [kN/m²]（任意）
level2:                           # 要求側と判定基準は入力（港湾基準で要確認）
  directions: [+y, -y]
  gravity: {D: 1.0, Ls: 1.0}      # 先に載せる長期荷重（係数は 1.0 のみ）
  kh: 0.25                        # 照査用震度（または target_displacement: 要求変位 [m]）
  max_displacement: 0.6           # 解析を打ち切る上部工の変位 [m]
  step: 0.005                     # 1ステップの変位増分の上限 [m]
  post_yield_ratio: 0.001
  inground_depth: 3.0             # 地中部のヒンジを置く範囲（海底面から 3/β）
  criteria: [no_mechanism, no_inground_plastic, {pile_rotation: 0.02}]
```
```python
r = run_level2(pier, "+y")        # r.print(), r.ok, r.curve, r.demand, r.criteria, r.po（PushoverResult）
```

- **地盤ばねモデル**: 海底面より下の杭を分割し、各節点に水平地盤ばね k = kh·D·（負担長）を X・Y に配置。
  杭先端は鉛直・ねじりを拘束。P1〜P3 の解析・照査もこのモデルで行える（杭の照査は地中部の各分割点も対象）。
  杭頭固定の杭頭変位が Chang の解に一致することをテストで確認
- **杭の塑性ヒンジ**: 全塑性モーメント Mp(N) = Zp·fy·cos(π|N|/2Ny)（腐食後の断面、Zp = (Do³ − Di³)/6）の
  バイリニア。N は長期荷重時の値で固定。位置は杭頭と、仮想固定点（仮想固定点モデル）または海底面〜地中部の
  各分割点（地盤ばねモデル）。載荷方向の曲げ面のみ。履歴は移動硬化のバイリニア（hinge.BilinearHinge、
  除荷は 2Mp の幅を剛のまま戻る）
- **地盤ばね**: pu を与えると弾完全塑性（上限 = pu·D·負担長）、なければ線形
- **載荷**: 長期荷重の後、上部工の重心に水平力（荷重係数 = 換算震度 kh = 水平力/W）を比例増加。
  事象追跡型の増分解析（建物編と同じ）
- **判定**: 要求点（kh または要求変位）で、崩壊機構が形成されない・地中部（海底面を含む）が全塑性に達しない・
  杭頭の塑性回転角が上限以下、から選択。上部工の梁は弾性とし、要求点で M ≦ Mu（材料係数 1.0）を確認（超えれば警告）

**P2 の簡略化**: 張出し部の荷重は端の梁への線荷重（ねじりは無視）。防舷材・係船柱の力は海側の梁の両端節点へ
てこの原理で配分（梁の局所的な曲げの影響は無視）。杭の慣性力・付加質量、車両・荷役機械の集中荷重の自動生成は
扱わない（集中荷重は loads の nodes / beams で入力）

**モデル化**
- 上部工: 縦梁・横梁の格子を model_level に配置し、上部工全体を 1つの剛床（マスター節点は平面の図心）
- 鋼管杭: 杭頭から仮想固定点までを 1本の部材とし、仮想固定点で固定。杭頭は上部工に剛結し、
  model_level から soffit までを剛域とする
- 仮想固定点 = 海底面 − 1/β、β = (kh·D / 4EI)^(1/4)（Chang の方法）。EI は解析の剛性に用いる断面
- 鋼管: A = π(D² − d²)/4、I = π(D⁴ − d⁴)/64、J = 2I、せん断の有効断面積 A/2。
  解析の剛性は腐食前（既定）、照査（P3）は腐食後の断面を用いる
- 上部工の梁の端部に剛域は設けない

## 利用可否シナリオ（直杭式横桟橋）
能登半島地震の被災を踏まえたガイドラインを参考に、地震後の桟橋の利用可否を多数のシナリオで検討する。
実施設のデータ（図面・FLIP の結果・船舶諸元）はないため、入力はパラメータとして与える。

| 段階 | 内容 | 状態 |
|---|---|---|
| U1 | 除荷・状態の引き継ぎ・変位制御（残留変位、被災後の利用荷重） | 済 |
| U2 | 地盤変位の入力（側方流動による地盤ばねの強制変位） | 済 |
| U3 | 鋼管杭のファイバーモデルによる M–φ（局部座屈ひずみ εcu） | 未 |
| U4 | 一括実行（仮想固定点モデルでのふるい分け → 地盤ばねモデルで詳細化） | 未 |
| U5 | 許容荷重と超過確率 | 未 |
| U6 | 判定カード（HTML） | 未 |

**U1: 増分解析の状態の引き継ぎ（pushover.Incremental）**
```python
from rcframe.pushover import Incremental, residual_table
inc = Incremental(model, hinges, levels, d, lateral_total=W)   # node_springs も可
inc.run("長期", lam_target=1.0)                                 # 鉛直荷重（cases の荷重ケース）
inc.run("地震", pattern=[(nid, P)], max_drift=0.02, lateral=True)
inc.unload([(nid, P)])                                         # 水平力を 0 まで除荷 → 残留変位
inc.run("利用", cases=["V"], lam_target=1.0)                   # 被災状態のまま別の荷重を載せる
rows = residual_table(inc, [(nid, P)], peaks=[0.005, 0.01])    # 最大点ごとの残留変位の表
```
- `run` は荷重係数（`lam_target`）・層間変形角（`max_drift`）・節点変位（`disp_target=(nid, dof, 値)`）の
  いずれかで止める（いずれもなければ ValueError）。ばね（材端ばね・地盤ばね）の状態は呼び出しの間で保たれる
- `snapshot()` / `restore()` で状態を保存・復元する（1本の載荷経路から複数の最大点の残留値を得る）
- 除荷では材端ばねは剛なので部材は弾性的に戻り、残留変位 = δmax − Pmax/K（弾性）。テストで照合
- 従来の `pushover(...)` は Incremental の薄い包みで、結果は変わらない

**U2: 側方流動（rcframe/port_flow.py）**
```yaml
lateral_flow:                       # 地盤ばねモデル（piles.model: soil_spring）が必要
  流動_海側1m:
    direction: +y
    surface: {L: 1.0, M: 0.8, S: 0.6}   # 海底面の地盤変位 ug0 [m]（列ごと、または全列共通の値）
    depth: 8.0                      # 流動する層の厚さ H [m]
    shape: cos                      # cos / linear / uniform
    kh_factor: 0.1                  # 流動層の地盤ばね剛性の低減係数（pu_factor: 上限の低減係数）
    kh: 0.15                        # 慣性力の換算震度（0 なら地盤変位のみ、負は逆向き）
    order: simultaneous             # simultaneous / ground_first / inertia_first
    unload: true                    # 最後に慣性力を除荷
```
```python
for case in pier.lateral_flow:
    r = run_flow(pier, case)        # r.print(), r.deck（peak / residual）, r.yielded, r.max_rotation("地中部")
```
- 地盤ばねの地盤側の端に強制変位 ug(ζ) を与える。ばねの力 = k·(u杭 − ug)、剛性方程式では等価節点力 k·Δug
  （`Incremental.run(ground=[(節点, 自由度, ug)])`）。上限 pu があれば杭に作用する力は pu·D·負担長 で頭打ち
- 分布: cos ug0·cos(πζ/2H)、linear ug0(1 − ζ/H)、uniform ug0（ζ: 海底面からの深さ、ζ > H で 0）
- 地中部の塑性ヒンジは全ての分割点に置く（流動層の下端付近の全塑性を捉えるため）
- 打ち切り: 上部工の変位が level2.max_displacement（地盤変位を含む段階は + 地盤変位の最大値）に達したら止める
- 出力: 上部工の最大・残留変位と平面の残留回転、杭頭・地中部の塑性回転角（最大・残留）、地盤ばねの降伏
- テスト: 自由な杭の剛体移動、杭頭固定の杭（Chang の解 H = 4EIβ³·ug、M0 = H/2β）、両端固定の杭に地盤反力の上限が
  作用する場合の端モーメント、地盤変位 0 で P4 と一致、弾性範囲で載荷順序によらない、上部工の水平方向の釣合い

**決定事項**: 鋼管の局部座屈ひずみ εcu は 式A 1.14/(D/t) − 0.012、式B 0.44·t/R − 0.0026、直接入力から選択
（下限 εy、D/t > 100 で警告）。塑性ヒンジ長 Lp = D（既定）、θu = (φu − φy)·Lp。
式A と式B の値の差（約 5 倍）は基準の原文で確認中

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
| 6 | 保有水平耐力（Ds・Fes、崩壊形の判定） | 済 |
| 7 | 3D可視化・計算書出力 | 済 |

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
- 保有水平耐力: 耐力壁のある建物（βu による Ds）は未対応。FD の部材を除いても崩壊形が保たれる場合の
  扱いはせず、FD の柱があれば部材群 D（安全側）。局部崩壊形の判定はしない。
  外力分布は Ai 分布で一定。偏心率の D 値は一次設計の Ex, Ey（重心のずれによるねじれを含む）から算定
