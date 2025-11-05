# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow もくもく会 - 初学者向けチュートリアル
# MAGIC
# MAGIC このノートブックでは、**MLflowを使わない場合の課題**を体験した後、**MLflowを使った解決方法**を学びます。
# MAGIC
# MAGIC ## 本日の流れ
# MAGIC 1. **データの準備とEDA**: データを理解する
# MAGIC 2. **Part 1**: MLflowなしで実験（課題を体験）
# MAGIC 3. **Part 2**: MLflowで同じ実験（課題解決を実感）
# MAGIC 4. **Part 3**: 複数モデルの比較（MLflowの真価を発揮）
# MAGIC 5. **Part 4**: モデルの再利用（実務での活用）
# MAGIC 6. **Part 5**: Unity Catalogへのモデル登録（組織での共有）
# MAGIC
# MAGIC ※ Databricks Free Edition (Serverless Compute) で動作確認済みです
# MAGIC
# MAGIC 📖 参考リンク：[Databricks 機械学習ガイド（日本語）](https://docs.databricks.com/ja/machine-learning/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 環境セットアップ
# MAGIC
# MAGIC 必要なライブラリをインストールします。
# MAGIC - **MLflow**: 機械学習の実験管理ツール
# MAGIC - **Plotly**: インタラクティブなグラフ作成ライブラリ
# MAGIC - すでにインストール済みの場合は「Requirement already satisfied」と表示されます
# MAGIC
# MAGIC 📖 参考リンク：
# MAGIC - [MLflow公式ドキュメント（英語）](https://mlflow.org/docs/latest/index.html)
# MAGIC - [Databricks MLflowガイド（日本語）](https://docs.databricks.com/ja/mlflow/index.html)

# COMMAND ----------

# MLflowとplotlyのインストール（最新版）と、Pythonカーネルの再起動
%pip install mlflow plotly
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 事前準備：データの読み込み
# MAGIC
# MAGIC 今回使用するのは **Diabetes データセット** です。
# MAGIC - 糖尿病患者のデータ（年齢、BMI、血圧など10個の特徴量）
# MAGIC - 1年後の病状進行度を予測する回帰問題
# MAGIC - 442サンプル
# MAGIC
# MAGIC **注意**: このデータセットの特徴量は**標準化済み**です
# MAGIC - 各特徴量は平均0、標準偏差1に変換されています
# MAGIC - そのため年齢（age）も小数点の値になっています
# MAGIC - 機械学習では、異なるスケールの特徴量を比較しやすくするため、このような前処理が一般的です
# MAGIC
# MAGIC 📖 参考リンク：[scikit-learn Diabetes データセット（英語）](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# データの読み込み
X, y = datasets.load_diabetes(return_X_y=True)

# データの分割（全実験で共通使用）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# データの確認
print(f"訓練データ: {X_train.shape[0]}サンプル")
print(f"テストデータ: {X_test.shape[0]}サンプル")
print(f"特徴量の数: {X_train.shape[1]}個")

# データフレームとして表示（最初の5行）
feature_names = datasets.load_diabetes().feature_names
df_sample = pd.DataFrame(X_train[:5], columns=feature_names)
df_sample['target'] = y_train[:5]

print("サンプルデータ（標準化済み）:")
display(df_sample)

print("\n特徴量の説明:")
print("age: 年齢, sex: 性別, bmi: BMI, bp: 血圧")
print("s1-s6: 血清測定値（コレステロール、血糖値など）")

# COMMAND ----------

# MAGIC %md
# MAGIC # 簡単なデータ探索（EDA）
# MAGIC
# MAGIC モデルを作る前に、データの特徴を理解しましょう
# MAGIC
# MAGIC 📖 参考リンク：[探索的データ解析（日本語）](https://docs.databricks.com/ja/exploratory-data-analysis/index.html)

# COMMAND ----------

# 基本統計量の確認
print("=== 基本統計量 ===")
df_all = pd.DataFrame(X, columns=feature_names)
df_all['target'] = y
display(df_all.describe())

# COMMAND ----------

# ターゲット変数（病状進行度）の分布を確認
import plotly.express as px

fig = px.histogram(
    df_all,
    x='target',
    nbins=30,
    title='ターゲット変数（病状進行度）の分布',
    labels={'target': '病状進行度', 'count': '患者数'},
    color_discrete_sequence=['skyblue']
)
fig.update_layout(showlegend=False)
fig.show()

print(f"最小値: {df_all['target'].min():.1f}")
print(f"最大値: {df_all['target'].max():.1f}")
print(f"平均値: {df_all['target'].mean():.1f}")
print(f"中央値: {df_all['target'].median():.1f}")

# COMMAND ----------

# 特徴量とターゲットの相関関係を確認
import plotly.graph_objects as go

# 相関係数を計算
correlations = df_all.corr()['target'].drop('target').sort_values(ascending=False)

# 棒グラフで表示
fig = go.Figure(data=[
    go.Bar(
        x=correlations.index,
        y=correlations.values,
        marker_color=['red' if v > 0 else 'blue' for v in correlations.values]
    )
])

fig.update_layout(
    title='各特徴量とターゲット変数の相関係数',
    xaxis_title='特徴量',
    yaxis_title='相関係数',
    height=400
)
fig.show()

print("\n=== 相関が強い特徴量トップ3 ===")
for i, (feature, corr) in enumerate(correlations.head(3).items(), 1):
    print(f"{i}. {feature}: {corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDAから分かったこと
# MAGIC
# MAGIC - ターゲット変数（病状進行度）は約25〜346の範囲に分布
# MAGIC - 特徴量の中で、**bmi（BMI）**や**s5**が病状進行度と比較的強い相関がある
# MAGIC - これらの特徴量が予測に重要な役割を果たす可能性が高い
# MAGIC
# MAGIC それでは、これらのデータを使ってモデルを作っていきましょう！

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: MLflowを使わない実験 😰
# MAGIC
# MAGIC まず、MLflowを使わずに機械学習モデルを作ってみましょう。
# MAGIC
# MAGIC ## やりたいこと
# MAGIC - 線形回帰モデルで予測
# MAGIC - 性能を評価
# MAGIC
# MAGIC ## 課題を体験してみましょう！

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# 実験開始（時刻を記録）
experiment_time = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"実験開始時刻: {experiment_time}")

# モデルの作成と学習
model = LinearRegression()
model.fit(X_train, y_train)

# 予測と評価
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n=== 実験結果 ===")
print(f"MSE (平均二乗誤差): {mse:.2f}")
print(f"R² スコア: {r2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 😰 MLflowなしの課題
# MAGIC
# MAGIC 上の実験で、以下の課題があります：
# MAGIC
# MAGIC ### 1. 実験結果が残らない
# MAGIC - セルを再実行したら、前の結果が消えてしまう
# MAGIC - 「さっきの実験、MSEいくつだったっけ？」→ わからない...
# MAGIC
# MAGIC ### 2. 複数実験の比較が大変
# MAGIC - 3つのモデルを試したとき、どれが良かった？
# MAGIC - コピペでメモ？エクセルに手入力？
# MAGIC
# MAGIC ### 3. 再現できない
# MAGIC - 「あの良い結果、どの設定でやったんだっけ？」
# MAGIC - モデルファイルはどこに保存した？
# MAGIC
# MAGIC **次のPart 2では、MLflowでこれらの課題を解決します！**

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: MLflowで同じ実験 😊
# MAGIC
# MAGIC 全く同じ実験を、今度は**MLflowを使って**やってみましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowの基本用語（5つだけ覚えましょう！）
# MAGIC
# MAGIC | 用語 | 意味 | 例 |
# MAGIC |------|------|-----|
# MAGIC | **Run（ラン）** | 1回の実験 | 「今日の1回目の実験」 |
# MAGIC | **Parameter（パラメータ）** | モデルの設定値 | test_size=0.2 |
# MAGIC | **Metric（メトリクス）** | 評価指標（数値） | MSE=2800.5 |
# MAGIC | **Model（モデル）** | 学習済みモデル | 保存されたファイル |
# MAGIC | **Experiment（エクスペリメント）** | 複数Runをまとめる箱 | 「糖尿病予測プロジェクト」 |
# MAGIC
# MAGIC 📖 参考リンク：[MLflow Tracking（日本語）](https://docs.databricks.com/ja/mlflow/tracking.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## エクスペリメントの種類
# MAGIC
# MAGIC MLflowには2種類のエクスペリメントがあります：
# MAGIC
# MAGIC ### 1. ノートブックエクスペリメント（今回使用）
# MAGIC - **ノートブックに自動的に添付**される
# MAGIC - ノートブックを開くと、右上のフラスコアイコンから確認できる
# MAGIC - 手軽で初心者におすすめ
# MAGIC - `mlflow.start_run()` を実行すると、自動的にノートブックエクスペリメントに記録される
# MAGIC
# MAGIC ### 2. ワークスペースエクスペリメント
# MAGIC - **ワークスペース内の独立したファイル**として作成
# MAGIC - 複数のノートブックから同じエクスペリメントを参照できる
# MAGIC - チームでの共有に便利
# MAGIC - `mlflow.set_experiment()` で明示的に指定が必要
# MAGIC
# MAGIC **今回はノートブックエクスペリメントを使用します！**
# MAGIC 特に設定しなくても、`mlflow.start_run()` を実行するだけで自動的に記録されるので簡単です。
# MAGIC
# MAGIC 📖 参考リンク：[エクスペリメントとラン（日本語）](https://docs.databricks.com/ja/mlflow/runs.html)

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# MLflowのランを開始（これだけ！）
# ノートブックエクスペリメントに自動的に記録されます
with mlflow.start_run(run_name="Linear Regression - with MLflow"):

    # --- ここから同じコード ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # --- ここまで同じ ---

    # MLflowに記録（追加部分）
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # モデルも保存
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train[:1]  # モデルの入力例
    )

    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    print("\n✅ MLflowに記録されました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 😊 MLflowで解決されたこと
# MAGIC
# MAGIC ### 1. 実験結果が自動保存される
# MAGIC - **右上のフラスコアイコン 🧪 をクリック** → 実験履歴が見える！
# MAGIC - MSE、R²、パラメータ、実行時刻、全て記録されている
# MAGIC
# MAGIC ### 2. コードで実験結果を取得できる
# MAGIC - 次のセルで実験履歴を確認してみましょう
# MAGIC
# MAGIC 📖 参考リンク：[MLflowでのモデルの記録（日本語）](https://docs.databricks.com/ja/mlflow/models.html)

# COMMAND ----------

# 実験履歴の取得
runs_df = mlflow.search_runs()

# run_nameカラムを作成（バージョン互換性のため）
if 'run_name' not in runs_df.columns:
    runs_df['run_name'] = runs_df['tags.mlflow.runName']

# 主要な情報だけ表示
display(runs_df[[
    'run_name',
    'start_time',
    'metrics.mse',
    'metrics.r2_score',
    'params.model_type'
]].sort_values('start_time', ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: 複数モデルの比較（MLflowの真価！）
# MAGIC
# MAGIC ここからがMLflowの本領発揮！
# MAGIC
# MAGIC **3つの異なるモデル**を試して、どれが一番良いか比較してみましょう：
# MAGIC 1. 線形回帰（Linear Regression）
# MAGIC 2. ランダムフォレスト（Random Forest）
# MAGIC 3. Gradient Boosting
# MAGIC
# MAGIC 📖 参考リンク：[実験の比較（日本語）](https://docs.databricks.com/ja/mlflow/tracking.html#compare-runs)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 実験するモデルのリスト
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# 各モデルで実験を実行
for model_name, model in models:
    with mlflow.start_run(run_name=model_name):

        # 学習
        model.fit(X_train, y_train)

        # 予測
        predictions = model.predict(X_test)

        # 評価
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)

        # MLflowに記録
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # モデル保存
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train[:1]
        )

        print(f"✅ {model_name}: MSE={mse:.2f}, R²={r2:.3f}")

print("\n🎉 全モデルの実験完了！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 結果を比較してみましょう！
# MAGIC
# MAGIC MLflowなら、複数実験を簡単に比較できます

# COMMAND ----------

# 全実験の取得
all_runs = mlflow.search_runs()

# run_nameカラムを作成（バージョン互換性のため）
if 'run_name' not in all_runs.columns:
    all_runs['run_name'] = all_runs['tags.mlflow.runName']

# 比較表を作成
comparison_df = all_runs[[
    'run_name',
    'metrics.mse',
    'metrics.rmse',
    'metrics.r2_score'
]].sort_values('metrics.mse')

print("=== モデル性能比較（MSEが小さいほど良い）===")
display(comparison_df)

# ベストモデルを表示
best_run = comparison_df.iloc[0]
print(f"\n🏆 ベストモデル: {best_run['run_name']}")
print(f"   MSE: {best_run['metrics.mse']:.2f}")
print(f"   R² Score: {best_run['metrics.r2_score']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 視覚化してみましょう

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# データの準備
comparison_df_sorted = comparison_df.sort_values('metrics.mse', ascending=True)
comparison_df_r2 = comparison_df.sort_values('metrics.r2_score', ascending=True)

# サブプロットの作成
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('モデル性能比較 - MSE', 'モデル性能比較 - R² Score')
)

# グラフ1: MSE比較
fig.add_trace(
    go.Bar(
        x=comparison_df_sorted['metrics.mse'],
        y=comparison_df_sorted['run_name'],
        orientation='h',
        marker_color='skyblue',
        name='MSE'
    ),
    row=1, col=1
)

# グラフ2: R²スコア比較
fig.add_trace(
    go.Bar(
        x=comparison_df_r2['metrics.r2_score'],
        y=comparison_df_r2['run_name'],
        orientation='h',
        marker_color='lightgreen',
        name='R² Score'
    ),
    row=1, col=2
)

# レイアウトの設定
fig.update_xaxes(title_text='MSE (小さいほど良い)', row=1, col=1)
fig.update_xaxes(title_text='R² Score (大きいほど良い)', row=1, col=2)
fig.update_layout(
    height=400,
    showlegend=False,
    title_text='モデル性能の比較'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: モデルの再利用（実務での活用）
# MAGIC
# MAGIC MLflowに保存したモデルは、いつでもロードして使えます！
# MAGIC
# MAGIC **想定シーン**：
# MAGIC - 「先週の実験で作った良いモデル、本番で使いたい」
# MAGIC - 「新しいデータで予測したい」
# MAGIC
# MAGIC 📖 参考リンク：[モデルのロードと推論（日本語）](https://docs.databricks.com/ja/mlflow/load-model.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 最も性能の良いモデルをロードする

# COMMAND ----------

# 最良のランを取得
best_run_info = mlflow.search_runs(
    order_by=["metrics.mse ASC"]
).iloc[0]

best_run_id = best_run_info['run_id']
# run_nameカラムを取得（バージョン互換性のため）
best_model_name = best_run_info.get('run_name', best_run_info.get('tags.mlflow.runName', 'Unknown'))

print(f"ロードするモデル: {best_model_name}")
print(f"Run ID: {best_run_id}")

# モデルのロード
model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print("\n✅ モデルのロード完了！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ロードしたモデルで予測してみる

# COMMAND ----------

# テストデータの一部で予測
sample_data = X_test[:5]
predictions = loaded_model.predict(sample_data)

# 結果を表示
results_df = pd.DataFrame({
    '予測値': predictions,
    '実際の値': y_test[:5],
    '誤差': np.abs(predictions - y_test[:5])
})

print("=== 予測結果 ===")
display(results_df)

print(f"\n平均誤差: {results_df['誤差'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Unity Catalogへのモデル登録
# MAGIC
# MAGIC ベストモデルを**Unity Catalog**に登録して、組織全体で管理・共有できるようにします。
# MAGIC
# MAGIC ## Unity Catalogとは？
# MAGIC - Databricksのデータとモデルのガバナンス機能
# MAGIC - モデルのバージョン管理、アクセス制御、系譜追跡が可能
# MAGIC - Databricks Free Editionでは `workspace.default` スキーマが最初から利用可能
# MAGIC
# MAGIC 📖 参考リンク：[Unity Catalogモデルレジストリ（日本語）](https://docs.databricks.com/ja/mlflow/model-registry.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベストモデルをUnity Catalogに登録

# COMMAND ----------

# ベストモデルの情報を再取得
best_run_info = mlflow.search_runs(
    order_by=["metrics.mse ASC"]
).iloc[0]

best_run_id = best_run_info['run_id']
best_model_name = best_run_info.get('run_name', best_run_info.get('tags.mlflow.runName', 'Unknown'))

print(f"登録するモデル: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"MSE: {best_run_info['metrics.mse']:.2f}")
print(f"R² Score: {best_run_info['metrics.r2_score']:.3f}")

# Unity Catalogのモデル名を設定
# フォーマット: カタログ名.スキーマ名.モデル名
uc_model_name = "workspace.default.diabetes_prediction_model"

print(f"\nUnity Catalogモデル名: {uc_model_name}")

# COMMAND ----------

# ベストモデルをUnity Catalogに登録
model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=uc_model_name
)

print(f"✅ モデルをUnity Catalogに登録しました！")
print(f"モデル名: {uc_model_name}")
print(f"バージョン: {model_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 登録されたモデルの確認
# MAGIC
# MAGIC Unity Catalogに登録したモデルは、以下の方法で確認できます：
# MAGIC 1. **カタログエクスプローラー**: 左サイドバーの「Catalog」から `workspace` → `default` → モデル名
# MAGIC 2. **コード**: MLflow APIを使用

# COMMAND ----------

# 登録されたモデルの情報を取得
from mlflow import MlflowClient

client = MlflowClient()

# モデルの最新情報を取得
model_info = client.get_registered_model(uc_model_name)

print(f"=== モデル情報 ===")
print(f"モデル名: {model_info.name}")
print(f"作成日時: {model_info.creation_timestamp}")
print(f"最終更新: {model_info.last_updated_timestamp}")

# すべてのバージョンを表示
model_versions = client.search_model_versions(f"name='{uc_model_name}'")
print(f"\n登録されているバージョン数: {len(model_versions)}")
for mv in model_versions:
    print(f"  - バージョン {mv.version}: {mv.current_stage} (Run ID: {mv.run_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalogからモデルをロードして使用

# COMMAND ----------

# Unity Catalogからモデルをロード
# バージョンを指定しない場合、最新バージョンが使用されます
uc_loaded_model = mlflow.pyfunc.load_model(f"models:/{uc_model_name}/{model_version.version}")

# 予測を実行
sample_predictions = uc_loaded_model.predict(X_test[:3])

# 結果を表示
uc_results_df = pd.DataFrame({
    '予測値': sample_predictions,
    '実際の値': y_test[:3],
    '誤差': np.abs(sample_predictions - y_test[:3])
})

print("=== Unity Catalogモデルでの予測結果 ===")
display(uc_results_df)

print("\n✅ Unity Catalogに登録したモデルで予測できました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalogのメリット
# MAGIC
# MAGIC ### ✅ 一元管理
# MAGIC - すべてのモデルが1箇所で管理される
# MAGIC - カタログエクスプローラーから簡単にアクセス
# MAGIC
# MAGIC ### ✅ バージョン管理
# MAGIC - モデルの全バージョンが保存される
# MAGIC - いつでも過去のバージョンに戻せる
# MAGIC
# MAGIC ### ✅ アクセス制御
# MAGIC - 誰がモデルを使用できるかを制御
# MAGIC - 本番環境での誤使用を防止
# MAGIC
# MAGIC ### ✅ 系譜追跡（リネージ）
# MAGIC - モデルがどのデータから作られたかを追跡
# MAGIC - ガバナンスとコンプライアンスに対応

# COMMAND ----------

# MAGIC %md
# MAGIC # まとめ：MLflowのメリット
# MAGIC
# MAGIC ## 今日体験したこと
# MAGIC
# MAGIC ### ✅ 実験管理が楽に
# MAGIC - 実験結果が自動保存される
# MAGIC - いつでも過去の実験を確認できる
# MAGIC - 「あれ、どの設定だっけ？」が無くなる
# MAGIC
# MAGIC ### ✅ 比較が簡単
# MAGIC - 複数モデルの性能を一目で比較
# MAGIC - ベストモデルがすぐわかる
# MAGIC - グラフ化も簡単
# MAGIC
# MAGIC ### ✅ 再現性の確保
# MAGIC - モデルを保存・ロード
# MAGIC - パラメータも全て記録
# MAGIC - 「同じ結果が出ない」が無くなる
# MAGIC
# MAGIC ### ✅ チームでの共有
# MAGIC - 実験結果をチームで共有
# MAGIC - 「私のマシンでは動く」問題の解消
# MAGIC
# MAGIC ### ✅ Unity Catalogで組織全体での管理
# MAGIC - モデルを一元管理
# MAGIC - バージョン管理とアクセス制御
# MAGIC - ガバナンスとコンプライアンスに対応
# MAGIC
# MAGIC ## 次のステップ
# MAGIC
# MAGIC - **ハイパーパラメータチューニング**: [Hyperoptの活用（日本語）](https://docs.databricks.com/ja/machine-learning/automl-hyperparam-tuning/index.html)
# MAGIC - **より複雑なモデルの管理**: [MLflow Projects（英語）](https://mlflow.org/docs/latest/projects.html)
# MAGIC - **モデルのデプロイ**: [モデルサービング（日本語）](https://docs.databricks.com/ja/machine-learning/model-serving/index.html)
# MAGIC - **MLflow Models Registry の活用**: [モデルレジストリ（日本語）](https://docs.databricks.com/ja/mlflow/model-registry.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 補足：よくある質問
# MAGIC
# MAGIC **Q1: MLflowはDatabricksでしか使えない？**
# MAGIC
# MAGIC A: いいえ、ローカル環境でも使えます！`pip install mlflow` でインストールできます。
# MAGIC
# MAGIC **Q2: MLflowのデータはどこに保存される？**
# MAGIC
# MAGIC A: Databricksでは自動的にワークスペースに保存されます。ローカルでは `mlruns` フォルダに保存されます。
# MAGIC
# MAGIC **Q3: 実験が増えすぎたら重くならない？**
# MAGIC
# MAGIC A: MLflowは効率的に管理するので大丈夫です。不要な実験は削除することもできます。
# MAGIC
# MAGIC **Q4: もっと詳しく学ぶには？**
# MAGIC
# MAGIC A:
# MAGIC - [MLflow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
# MAGIC - [Databricks MLflowガイド](https://docs.databricks.com/mlflow/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎉 お疲れ様でした！
# MAGIC
# MAGIC MLflowの基本的な使い方と、そのメリットを体験していただけたでしょうか？
# MAGIC
# MAGIC 今日学んだことを、ぜひ自分のプロジェクトでも試してみてください！
