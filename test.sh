#!/bin/bash
set -e

# ディレクトリ設定
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR/tinygrad"

echo "=========================================================="
echo "   Nano-Mirage vs Tinygrad(Original) 性能比較ベンチマーク"
echo "=========================================================="
echo "このスクリプトは、Metalバックエンドにおける「ホスト側制御ロジック」の"
echo "高速化効果（Pythonオーバーヘッドの削減）を検証します。"
echo ""

# 1. 環境構築
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

PYTHON="$BASE_DIR/tinygrad/venv/bin/python"
PIP="$BASE_DIR/tinygrad/venv/bin/pip"

# echo "Installing dependencies..."
# $PIP install -e . numpy matplotlib > /dev/null

# 2. ファイル準備
GRAPH_DIR="tinygrad/runtime/graph"
METAL_PY="$GRAPH_DIR/metal.py"
METAL_OLD="$GRAPH_DIR/metal_old.py"

if [ ! -f "$METAL_OLD" ]; then
    echo "Creating baseline (slow) version as metal_old.py..."
    cp "$METAL_PY" "${METAL_PY}.fast.bak"
    git checkout "$METAL_PY" || true
    cp "$METAL_PY" "$METAL_OLD"
    mv "${METAL_PY}.fast.bak" "$METAL_PY"
fi

# 3. ベンチマーク実行
export METAL=1

# シナリオ1: 軽量モデル
echo ""
echo "----------------------------------------------------------"
echo "【シナリオ1】 軽量モデル (Small)"
echo "----------------------------------------------------------"
echo "GPUの計算負荷が低い（数ミリ秒以下）モデルです。"
echo "「Pythonの遅さ」が支配的になるため、高速化の効果が劇的に出ます。"
echo "TinyMLやリアルタイム制御、強化学習などの用途を想定しています。"
$PYTHON benchmark_large.py --mode=plot --size=small

# シナリオ2: 大規模モデル
echo ""
echo "----------------------------------------------------------"
echo "【シナリオ2】 大規模モデル (Large)"
echo "----------------------------------------------------------"
echo "Llama-2 13Bクラスの重厚な計算モデルです。"
echo "GPUの計算時間（数十ミリ秒）が支配的になるため、ホスト側の高速化効果は"
echo "相対的に小さく見えます（誤差に埋もれる場合があります）。"
$PYTHON benchmark_large.py --mode=plot --size=large

echo ""
echo "=========================================================="
echo "✅ 全テスト完了"
echo "結果グラフ:"
echo "  1. tinygrad/benchmark_result_small.png (効果: 大)"
echo "  2. tinygrad/benchmark_result_large.png (効果: 小)"
echo "=========================================================="
