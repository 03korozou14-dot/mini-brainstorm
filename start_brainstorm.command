#!/bin/zsh
# Mini Idobata Brainstorm サーバー起動スクリプト
# Finder からダブルクリックするだけで
# - 仮想環境の有効化
# - サーバー起動
# - キー入力での安全な停止
# を行う。

# このスクリプト自身が置かれている mini_brainstorm ディレクトリに移動
cd "$(dirname "$0")" || exit 1

# 仮想環境を有効化
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo ".venv が見つかりません。先に python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt を実行してください。"
  read -n 1 -s -r -p "何かキーを押すとウィンドウを閉じます..."
  exit 1
fi

# すでに 8000 番ポートで動いているサーバーがあれば停止
if lsof -ti :8000 >/dev/null 2>&1; then
  echo "既存のサーバープロセスを停止します (port 8000)..."
  kill $(lsof -ti :8000) 2>/dev/null || true
  # 完全に落ちるまで少し待つ
  sleep 1
fi

echo "Mini Idobata Brainstorm サーバーを起動します..."
python app/main.py

# サーバー停止用の一時停止（サーバーが終了したあとにだけ実行される）
read -n 1 -s -r -p "サーバーを停止しました。ウィンドウを閉じるには何かキーを押してください..."
