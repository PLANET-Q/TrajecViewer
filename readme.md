# Trajectory Viewer

ロケットの弾道表示GUIスクリプト
csv形式のシンプルな飛行履歴ファイルからインタラクティブに操作可能な弾道の3Dビューを表示することができます．

## Installation

```clone.sh
# clone
git clone https://github.com/PLANET-Q/TrajecViewer.git

# install dependencies
cd ./TrajecViewer
pip install -r requirements.txt
```

## Usage

## File formats

### Trajectory file format

読み取られる弾道履歴ファイルはカンマ( `,` )区切りのcsvファイルフォーマットです．
以下に示す形式で，時刻tに対する弾道履歴が保存されていることを想定します

```trajec.csv
t,x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
...
```

### Rocket parameter file format

ロケットのパラメータファイルは以下に示すjsonフォーマットです．
現在はロケット全長( `height` )，直径( `diameter` )，乾燥時重心位置( `CG_dry` )のみが使用されます．

```sample_parameter.json
{
    "name": "PQ_sample_parameter",
    "height": 4.768,
    "diameter": 0.167,
    "CG_dry": 3.072,
    "CP": 3.572,
    ...
}
```

### Additional Event Log file format

任意の追加パラメータとして，飛行時のイベントログのファイルを入力できます．
イベントログは以下に示すjsonファイルフォーマットです．

イベントログを読み込んだ場合，飛行中に起こったイベントがビューの弾道軌跡上に表示されます．
各キーバリュー がそれぞれ一つのイベントを表し，キーがビューに表示されるイベント名となります．
valueには少なくとも `"t"` キーが含まれている必要があり，これはそのイベントが発生した時刻を表します．
~~valueに `"t"` 以外のキーが含まれる場合，時刻 `t` においてそのキーが表す飛行パラメータを取得し，表示します．~~ ←開発中

```sample_event_log.json
{
    "1stlug_off": {
        "t": 0.8871310867300384
    },
    "2ndlug_off": {
        "t": 0.9277499738390353
    },
    "MECO": {
        "t": 20.002125297551054
    },
    "apogee": {
        "t": 36.50134000289616
    },
    "drogue": {
        "t": 37.572224892984146
    },
    "para": {
        "t": 174.0691701489237
    },
    "landing": {
        "t": 210.1275115979232
    },
    "MaxQ": {
        "Q": 67912.60811252156,
        "t": 19.98999999999964,
        "p": 68114.27212732563,
        "T": 276.32518343651003,
        "mach": 1.1934579578753168
    },
    "MaxMach": {
        "Q": 67912.60811252156,
        "t": 19.98999999999964,
        "p": 68114.27212732563,
        "T": 276.32518343651003,
        "mach": 1.1934579578753168
    },
    "MaxV": {
        "Q": 67912.60811252156,
        "t": 19.98999999999964,
        "p": 68114.27212732563,
        "T": 276.32518343651003,
        "speed": 397.7734446858006,
        "mach": 1.1934579578753168
    }
}
```

### Rocket 3D Model

弾道履歴として表示されるロケットの3Dモデルを指定する事ができます．現在は `.obj` フォーマットにのみ対応しています．
指定しない場合，デフォルトのobjファイル( `samples/std_scale.obj` ) が読み込まれ，表示されるモデルは，
ロケットパラメータファイルの全長 `height` , 直径 `diameter` に応じて伸張されます．

## Samples

`samples` フォルダにサンプルの飛行履歴ファイル，ロケットパラメータ，イベントログファイルがあります．

## Other

### Pre rendering

`config.json` はビューワプログラムの環境設定を記述したファイルです．

~~現在は `use_pre_rendering` キーで指定される プリレンダリング機能（beta）の有効/無効設定のみが利用可能です．
`use_pre_rendering` を `true` にすると，弾道読み込み時に，各時刻におけるモデルの頂点位置データを予め全て計算します．
これにより，Playbackモード（Startボタンを押して自動再生）のフレームレートを向上することを狙っています~~

→数フレーム毎に頂点データがうまく計算されないバグが発生，現在修正中です．

## TODO

- マッハ数等他の飛行中パラメータの表示
- Playbackモードにおける高精度タイマーの使用
- 飛行中の重心移動の実装
- 重心位置，圧力中心位置の表示
- 地図表示

## License

This program is licensed under the GPL-v3 License