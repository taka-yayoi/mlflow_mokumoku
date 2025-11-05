# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow もくもく会 - 生成AIのトレーシング & 評価
# MAGIC
# MAGIC このノートブックでは、**MLflowを使った生成AIアプリケーションのトレーシングと評価**を学びます。
# MAGIC
# MAGIC ## 本日の流れ
# MAGIC 1. **環境セットアップ**: 必要なライブラリのインストール
# MAGIC 2. **Part 1**: Tracingの基本とAuto-tracing
# MAGIC 3. **Part 2**: GenAI Evaluationの実践
# MAGIC 4. **Part 3**: Custom Judges APIの使用
# MAGIC 5. **Part 4**: 本番環境モニタリング
# MAGIC
# MAGIC ※ Databricks Free Edition (Serverless Compute) で動作確認済みです
# MAGIC
# MAGIC 📖 参考リンク：
# MAGIC - [MLflow Tracing（英語）](https://mlflow.org/docs/latest/llms/tracing/index.html)
# MAGIC - [MLflow LLM Evaluation（英語）](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 環境セットアップ
# MAGIC
# MAGIC 必要なライブラリをインストールします。
# MAGIC - **MLflow**: 実験管理とトレーシング
# MAGIC - **Databricks SDK**: Foundation Model APIアクセス
# MAGIC - **Pandas**: データ処理

# COMMAND ----------

# MLflowとDatabricks SDKのインストール
%pip install mlflow databricks-sdk pandas plotly
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Tracingの基本とAuto-tracing
# MAGIC
# MAGIC **トレーシング（Tracing）**とは、LLMアプリケーションの実行過程を記録・可視化する機能です。
# MAGIC
# MAGIC ## なぜトレーシングが必要？
# MAGIC
# MAGIC ### 😰 トレーシングなしの課題
# MAGIC - LLMへのプロンプトと応答が見えない
# MAGIC - エラーの原因が分からない
# MAGIC - レイテンシ（遅延）のボトルネックが不明
# MAGIC - コストが把握できない
# MAGIC
# MAGIC ### 😊 トレーシングのメリット
# MAGIC - すべてのLLM呼び出しが記録される
# MAGIC - プロンプトと応答が可視化される
# MAGIC - 実行時間とコストが追跡できる
# MAGIC - デバッグが簡単になる
# MAGIC
# MAGIC 📖 参考リンク：[MLflow Tracing入門（英語）](https://mlflow.org/docs/latest/llms/tracing/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Foundation Model APIの設定
# MAGIC
# MAGIC Databricks Foundation Model API (FMAPI) を使用してLLMを呼び出します。

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# WorkspaceClientの初期化
w = WorkspaceClient()

def call_fmapi(prompt: str, endpoint: str = "databricks-gpt-oss-20b") -> str:
    """
    Databricks Foundation Model APIを呼び出す

    Args:
        prompt: LLMへの入力プロンプト
        endpoint: Foundation Modelのエンドポイント名

    Returns:
        LLMからの応答テキスト
    """
    response = w.serving_endpoints.query(
        name=endpoint,
        messages=[
            ChatMessage(role=ChatMessageRole.USER, content=prompt)
        ]
    )
    return response.choices[0].message.content

print("✅ Foundation Model API設定完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## シンプルなLLM関数（トレーシングなし）

# COMMAND ----------

def simple_llm_call(question: str) -> str:
    """
    シンプルなLLM呼び出し（トレーシングなし）
    """
    # Databricks Foundation Model APIを使用
    return call_fmapi(question)

# トレーシングなしで実行
print("=== トレーシングなしの実行 ===")
result = simple_llm_call("MLflowとは何ですか？")
print(f"質問: MLflowとは何ですか？")
print(f"回答: {result}")
print("\n⚠️ 実行履歴が残りません！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-tracingの有効化
# MAGIC
# MAGIC MLflowの**Auto-tracing**を使うと、LLM呼び出しを自動的にトレースできます。

# COMMAND ----------

import mlflow

# Auto-tracingを有効化
mlflow.autolog()

# トレースされる関数
@mlflow.trace
def traced_llm_call(question: str) -> str:
    """
    トレースされるLLM呼び出し
    """
    # Databricks Foundation Model APIを使用
    return call_fmapi(question)

# トレーシングありで実行
print("=== トレーシングありの実行 ===")
result = traced_llm_call("トレーシングはどのように機能しますか？")
print(f"質問: トレーシングはどのように機能しますか？")
print(f"回答: {result}")
print("\n✅ トレースが記録されました！")
print("右側の「Traces」タブから確認できます")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 複数ステップのトレーシング
# MAGIC
# MAGIC RAG（Retrieval-Augmented Generation）のような複数ステップの処理もトレースできます。

# COMMAND ----------

@mlflow.trace(name="retrieve_documents")
def retrieve_documents(query: str) -> list:
    """
    ドキュメント検索（モック）
    """
    # モックドキュメント
    docs = [
        "MLflowは機械学習ワークフローを管理するためのオープンソースプラットフォームです。",
        "MLflowはトラッキング、プロジェクト、モデル、レジストリのコンポーネントを提供します。",
        "トレーシングはLLMアプリケーションのデバッグと最適化に役立ちます。"
    ]

    return docs

@mlflow.trace(name="generate_answer")
def generate_answer(query: str, context: list) -> str:
    """
    コンテキストを使って回答生成
    """
    # コンテキストを使ったプロンプトを作成
    context_text = "\n".join(context)
    prompt = f"""以下のコンテキストを参考に、質問に答えてください。

コンテキスト:
{context_text}

質問: {query}

回答:"""

    # Databricks Foundation Model APIを使用
    return call_fmapi(prompt)

@mlflow.trace(name="rag_pipeline")
def rag_pipeline(query: str) -> dict:
    """
    RAGパイプライン全体
    """
    # ステップ1: ドキュメント検索
    documents = retrieve_documents(query)

    # ステップ2: 回答生成
    answer = generate_answer(query, documents)

    return {
        "query": query,
        "documents": documents,
        "answer": answer
    }

# RAGパイプラインを実行
print("=== RAGパイプラインの実行 ===")
result = rag_pipeline("MLflowとは何ですか？")
print(f"質問: {result['query']}")
print(f"回答: {result['answer']}")
print("\n✅ 全ステップがトレースされました！")

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレースの確認
# MAGIC
# MAGIC トレース情報をプログラムで取得することもできます。

# COMMAND ----------

# 最新のトレースを取得
traces = mlflow.search_traces()

if len(traces) > 0:
    print("=== 最新のトレース情報 ===")
    print(f"トレース件数: {len(traces)}")

    # 基本情報のみを表示（Arrowエラーを回避）
    print("\n最新のトレース:")
    for i, row in traces.head(5).iterrows():
        print(f"\n--- トレース {i+1} ---")
        if 'request_id' in traces.columns:
            print(f"  リクエストID: {row.get('request_id', 'N/A')}")
        if 'trace_name' in traces.columns:
            print(f"  トレース名: {row.get('trace_name', 'N/A')}")
        if 'execution_time_ms' in traces.columns:
            exec_time = row.get('execution_time_ms', 0)
            if exec_time:
                print(f"  実行時間: {exec_time:.2f}ms")
        if 'status' in traces.columns:
            print(f"  ステータス: {row.get('status', 'N/A')}")
else:
    print("⚠️ トレースが見つかりません")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: GenAI Evaluationの実践
# MAGIC
# MAGIC LLMアプリケーションの品質を評価します。
# MAGIC
# MAGIC ## 評価の重要性
# MAGIC
# MAGIC LLMの出力は確率的なので、以下を評価する必要があります：
# MAGIC - **正確性（Accuracy）**: 回答が正しいか
# MAGIC - **関連性（Relevance）**: 質問に関連しているか
# MAGIC - **有害性（Toxicity）**: 有害なコンテンツを含まないか
# MAGIC - **幻覚（Hallucination）**: 事実と異なる情報を生成していないか
# MAGIC
# MAGIC 📖 参考リンク：[LLM Evaluation Guide（英語）](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価用のデータセットを準備

# COMMAND ----------

import pandas as pd

# 評価用のQAデータセット
eval_data = pd.DataFrame({
    "question": [
        "MLflowとは何ですか？",
        "MLflowのトラッキング機能はどのように動作しますか？",
        "モデルレジストリとは何ですか？",
        "MLflowを使ってモデルをデプロイする方法は？",
        "MLflow Projectsとは何ですか？"
    ],
    "ground_truth": [
        "MLflowは機械学習ライフサイクル全体を管理するためのオープンソースプラットフォームです。",
        "MLflow Trackingを使用すると、MLコードを実行する際にパラメータ、メトリクス、アーティファクトをログできます。",
        "モデルレジストリは、モデルのバージョンとライフサイクルを管理するための集中型モデルストアです。",
        "MLflow Modelsは標準フォーマットを使用して、さまざまなプラットフォームにデプロイできます。",
        "MLflow Projectsは、再利用可能で再現可能な形式でMLコードをパッケージ化します。"
    ]
})

print("=== 評価データセット ===")
display(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの予測を取得

# COMMAND ----------

def qa_model(question: str) -> str:
    """
    質問応答モデル
    """
    # Databricks Foundation Model APIを使用
    return call_fmapi(question)

# 予測を生成
eval_data["prediction"] = eval_data["question"].apply(qa_model)

print("=== 予測結果 ===")
display(eval_data[["question", "prediction"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 基本的な評価メトリクス
# MAGIC
# MAGIC MLflowの組み込み評価メトリクスを使います。

# COMMAND ----------

from mlflow.metrics import exact_match, token_count

# 評価の実行
with mlflow.start_run(run_name="QA Model Evaluation"):

    # メトリクスの計算
    for idx, row in eval_data.iterrows():
        # Exact Matchの計算
        is_exact_match = row["prediction"] == row["ground_truth"]
        mlflow.log_metric(f"exact_match_{idx}", int(is_exact_match))

        # トークン数の計算（簡易版）
        token_count_pred = len(row["prediction"].split())
        mlflow.log_metric(f"token_count_{idx}", token_count_pred)

    # 全体の統計
    accuracy = (eval_data["prediction"] == eval_data["ground_truth"]).mean()
    avg_token_count = eval_data["prediction"].apply(lambda x: len(x.split())).mean()

    mlflow.log_metric("overall_accuracy", accuracy)
    mlflow.log_metric("avg_token_count", avg_token_count)

    # 評価データを保存
    mlflow.log_table(eval_data, artifact_file="evaluation_results.json")

    print("=== 評価結果 ===")
    print(f"正確性: {accuracy:.2%}")
    print(f"平均トークン数: {avg_token_count:.1f}")
    print("\n✅ 評価結果がMLflowに記録されました！")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Custom Judges APIの使用
# MAGIC
# MAGIC **Judge**は、LLMの出力を評価するための評価者（別のLLMや関数）です。
# MAGIC
# MAGIC ## Judgeの種類
# MAGIC - **Built-in Judges**: MLflowが提供する標準的な評価者
# MAGIC - **Custom Judges**: 独自の評価ロジックを実装
# MAGIC
# MAGIC 📖 参考リンク：[Custom Metrics（英語）](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#custom-llm-evaluation-metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## カスタムJudgeの作成

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric

# カスタムJudge 1: 回答の長さを評価
def length_judge(predictions, targets=None, metrics=None):
    """
    回答の長さが適切かを評価
    """
    scores = []
    for pred in predictions:
        length = len(pred.split())
        # 10-50単語が適切と仮定
        if 10 <= length <= 50:
            score = 1.0
        elif length < 10:
            score = 0.5  # 短すぎる
        else:
            score = 0.7  # 長すぎるが許容
        scores.append(score)

    return {"scores": scores}

# カスタムJudge 2: キーワードの存在を確認
def keyword_judge(predictions, targets=None, metrics=None):
    """
    重要なキーワードが含まれているかを評価
    """
    important_keywords = ["MLflow", "model", "tracking", "registry", "platform"]
    scores = []

    for pred in predictions:
        pred_lower = pred.lower()
        keyword_count = sum(1 for kw in important_keywords if kw.lower() in pred_lower)
        score = min(keyword_count / 2.0, 1.0)  # 最大1.0
        scores.append(score)

    return {"scores": scores}

# カスタムJudge 3: 文の完全性を評価
def completeness_judge(predictions, targets=None, metrics=None):
    """
    文が完全に終わっているかを評価
    """
    scores = []
    for pred in predictions:
        # ピリオド、疑問符、感嘆符で終わっているか
        if pred.strip() and pred.strip()[-1] in '.!?':
            score = 1.0
        else:
            score = 0.5
        scores.append(score)

    return {"scores": scores}

print("✅ カスタムJudgeを3つ作成しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## カスタムJudgeで評価

# COMMAND ----------

with mlflow.start_run(run_name="Custom Judge Evaluation"):

    predictions = eval_data["prediction"].tolist()

    # 各Judgeで評価
    length_result = length_judge(predictions)
    keyword_result = keyword_judge(predictions)
    completeness_result = completeness_judge(predictions)

    # 結果をDataFrameに追加
    eval_data["length_score"] = length_result["scores"]
    eval_data["keyword_score"] = keyword_result["scores"]
    eval_data["completeness_score"] = completeness_result["scores"]

    # 総合スコアを計算
    eval_data["overall_score"] = (
        eval_data["length_score"] * 0.3 +
        eval_data["keyword_score"] * 0.4 +
        eval_data["completeness_score"] * 0.3
    )

    # 平均スコアをログ
    mlflow.log_metric("avg_length_score", eval_data["length_score"].mean())
    mlflow.log_metric("avg_keyword_score", eval_data["keyword_score"].mean())
    mlflow.log_metric("avg_completeness_score", eval_data["completeness_score"].mean())
    mlflow.log_metric("avg_overall_score", eval_data["overall_score"].mean())

    # 結果を保存
    mlflow.log_table(eval_data, artifact_file="custom_judge_results.json")

    print("=== カスタムJudge評価結果 ===")
    display(eval_data[["question", "length_score", "keyword_score", "completeness_score", "overall_score"]])

    print(f"\n平均スコア:")
    print(f"  長さ: {eval_data['length_score'].mean():.2f}")
    print(f"  キーワード: {eval_data['keyword_score'].mean():.2f}")
    print(f"  完全性: {eval_data['completeness_score'].mean():.2f}")
    print(f"  総合: {eval_data['overall_score'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## スコアの可視化

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# レーダーチャートで可視化
categories = ['長さ', 'キーワード', '完全性']
scores = [
    eval_data['length_score'].mean(),
    eval_data['keyword_score'].mean(),
    eval_data['completeness_score'].mean()
]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=scores,
    theta=categories,
    fill='toself',
    name='スコア'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=False,
    title="カスタムJudge評価スコア"
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: 本番環境モニタリング
# MAGIC
# MAGIC 本番環境でのLLMアプリケーションを継続的にモニタリングします。
# MAGIC
# MAGIC ## モニタリングの重要性
# MAGIC
# MAGIC ### なぜモニタリングが必要？
# MAGIC - **品質の低下検出**: モデルの出力品質が低下していないか
# MAGIC - **コスト管理**: トークン使用量とコストの追跡
# MAGIC - **パフォーマンス**: レイテンシと可用性の監視
# MAGIC - **ユーザーフィードバック**: 実際の使用状況の把握
# MAGIC
# MAGIC 📖 参考リンク：[MLflow Deployment（英語）](https://mlflow.org/docs/latest/deployment/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 本番トラフィックのシミュレーション

# COMMAND ----------

import random
from datetime import datetime, timedelta

# 本番トラフィックをシミュレート
def simulate_production_traffic(num_requests=20):
    """
    本番環境のリクエストをシミュレート
    """
    questions = [
        "MLflowとは何ですか？",
        "MLflowのトラッキングはどのように機能しますか？",
        "モデルレジストリとは何ですか？",
        "モデルをデプロイする方法は？",
        "MLflow Projectsとは何ですか？",
        "PythonでMLflowを使用する方法は？",
        "MLflowモデルとは何ですか？",
    ]

    traffic_data = []

    for i in range(num_requests):
        question = random.choice(questions)

        # トレースを記録
        with mlflow.start_run(run_name=f"production_request_{i}"):
            start_time = time.time()

            # 予測を実行
            prediction = qa_model(question)

            latency = time.time() - start_time
            token_count = len(prediction.split())

            # メトリクスを記録
            mlflow.log_param("question", question)
            mlflow.log_metric("latency_ms", latency * 1000)
            mlflow.log_metric("token_count", token_count)
            mlflow.log_metric("timestamp", time.time())

            # ランダムなフィードバックスコア（1-5）
            feedback_score = random.randint(3, 5)
            mlflow.log_metric("user_feedback", feedback_score)

            traffic_data.append({
                "request_id": i,
                "question": question,
                "prediction": prediction,
                "latency_ms": latency * 1000,
                "token_count": token_count,
                "user_feedback": feedback_score,
                "timestamp": datetime.now() - timedelta(minutes=num_requests - i)
            })

    return pd.DataFrame(traffic_data)

# トラフィックをシミュレート
print("=== 本番トラフィックをシミュレート中 ===")
production_data = simulate_production_traffic(20)
print(f"✅ {len(production_data)}件のリクエストを記録しました")

display(production_data.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## モニタリングダッシュボード

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ダッシュボードを作成
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'レイテンシの推移',
        'トークン使用量',
        'ユーザーフィードバック',
        'リクエスト数の推移'
    )
)

# グラフ1: レイテンシの推移
fig.add_trace(
    go.Scatter(
        x=production_data['request_id'],
        y=production_data['latency_ms'],
        mode='lines+markers',
        name='レイテンシ',
        line=dict(color='blue')
    ),
    row=1, col=1
)

# グラフ2: トークン使用量のヒストグラム
fig.add_trace(
    go.Histogram(
        x=production_data['token_count'],
        name='トークン数',
        marker_color='green'
    ),
    row=1, col=2
)

# グラフ3: ユーザーフィードバックの分布
feedback_counts = production_data['user_feedback'].value_counts().sort_index()
fig.add_trace(
    go.Bar(
        x=feedback_counts.index,
        y=feedback_counts.values,
        name='フィードバック',
        marker_color='orange'
    ),
    row=2, col=1
)

# グラフ4: 累積リクエスト数
fig.add_trace(
    go.Scatter(
        x=production_data['request_id'],
        y=range(1, len(production_data) + 1),
        mode='lines',
        name='累積リクエスト',
        line=dict(color='purple')
    ),
    row=2, col=2
)

# レイアウト設定
fig.update_xaxes(title_text="リクエストID", row=1, col=1)
fig.update_yaxes(title_text="レイテンシ (ms)", row=1, col=1)

fig.update_xaxes(title_text="トークン数", row=1, col=2)
fig.update_yaxes(title_text="頻度", row=1, col=2)

fig.update_xaxes(title_text="評価スコア", row=2, col=1)
fig.update_yaxes(title_text="件数", row=2, col=1)

fig.update_xaxes(title_text="リクエストID", row=2, col=2)
fig.update_yaxes(title_text="累積件数", row=2, col=2)

fig.update_layout(
    height=600,
    showlegend=False,
    title_text="本番環境モニタリングダッシュボード"
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モニタリング統計

# COMMAND ----------

# 統計情報を計算
stats = {
    "総リクエスト数": len(production_data),
    "平均レイテンシ (ms)": production_data['latency_ms'].mean(),
    "最大レイテンシ (ms)": production_data['latency_ms'].max(),
    "平均トークン数": production_data['token_count'].mean(),
    "総トークン数": production_data['token_count'].sum(),
    "平均ユーザー評価": production_data['user_feedback'].mean(),
}

print("=== 本番環境統計 ===")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# MLflowに記録
with mlflow.start_run(run_name="Production Monitoring Summary"):
    for key, value in stats.items():
        mlflow.log_metric(key.replace(" ", "_").replace("(", "").replace(")", ""), value)

    mlflow.log_table(production_data, artifact_file="production_traffic.json")

print("\n✅ モニタリングデータをMLflowに記録しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## アラート条件の設定

# COMMAND ----------

# アラート条件をチェック
def check_alerts(data):
    """
    アラート条件をチェック
    """
    alerts = []

    # レイテンシアラート（1000ms超過）
    high_latency = data[data['latency_ms'] > 1000]
    if len(high_latency) > 0:
        alerts.append(f"⚠️ 高レイテンシ検出: {len(high_latency)}件のリクエストが1秒以上")

    # 低評価アラート（スコア3以下）
    low_feedback = data[data['user_feedback'] <= 3]
    if len(low_feedback) > len(data) * 0.3:  # 30%以上
        alerts.append(f"⚠️ 低評価が多い: {len(low_feedback)}件（{len(low_feedback)/len(data)*100:.1f}%）")

    # トークン使用量アラート
    avg_tokens = data['token_count'].mean()
    if avg_tokens > 50:
        alerts.append(f"⚠️ トークン使用量が多い: 平均{avg_tokens:.1f}トークン")

    return alerts

# アラートチェック
alerts = check_alerts(production_data)

print("=== アラート状況 ===")
if alerts:
    for alert in alerts:
        print(alert)
else:
    print("✅ すべて正常です")

# COMMAND ----------

# MAGIC %md
# MAGIC # まとめ：生成AIトレーシング & 評価のメリット
# MAGIC
# MAGIC ## 今日体験したこと
# MAGIC
# MAGIC ### ✅ トレーシング
# MAGIC - LLM呼び出しの自動記録
# MAGIC - 複数ステップの処理を可視化
# MAGIC - デバッグとパフォーマンス最適化が容易
# MAGIC
# MAGIC ### ✅ 評価（Evaluation）
# MAGIC - 組み込みメトリクスで品質測定
# MAGIC - カスタムJudgeで独自の評価基準を実装
# MAGIC - 継続的な品質モニタリング
# MAGIC
# MAGIC ### ✅ 本番モニタリング
# MAGIC - リアルタイムのパフォーマンス追跡
# MAGIC - ユーザーフィードバックの収集
# MAGIC - アラート設定で問題を早期発見
# MAGIC
# MAGIC ## 次のステップ
# MAGIC
# MAGIC - **実際のLLM APIとの統合**: OpenAI、Anthropicなど
# MAGIC - **より高度な評価**: BLEU、ROUGE、BERTScoreなど
# MAGIC - **A/Bテスト**: 複数のプロンプトやモデルを比較
# MAGIC - **プロダクション展開**: MLflow Deploymentの活用
# MAGIC
# MAGIC 📖 参考リンク：
# MAGIC - [MLflow LLMsガイド（英語）](https://mlflow.org/docs/latest/llms/index.html)
# MAGIC - [Databricks LLMsガイド（日本語）](https://docs.databricks.com/ja/generative-ai/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 補足：よくある質問
# MAGIC
# MAGIC **Q1: トレーシングはコストがかかる？**
# MAGIC
# MAGIC A: トレーシング自体のオーバーヘッドは最小限です。LLM呼び出しのコストが主な部分です。
# MAGIC
# MAGIC **Q2: カスタムJudgeにLLMを使える？**
# MAGIC
# MAGIC A: はい！実際には別のLLMを使って出力を評価することが一般的です（LLM-as-a-Judge）。
# MAGIC
# MAGIC **Q3: 本番環境でトレーシングを無効化すべき？**
# MAGIC
# MAGIC A: サンプリング（一部のリクエストのみトレース）を使えば、本番でも有効にできます。
# MAGIC
# MAGIC **Q4: 複数のLLMプロバイダーに対応している？**
# MAGIC
# MAGIC A: はい！OpenAI、Anthropic、HuggingFace、Azure OpenAIなど多数対応しています。
# MAGIC
# MAGIC **Q5: もっと詳しく学ぶには？**
# MAGIC
# MAGIC A:
# MAGIC - [MLflow LLMs公式ドキュメント（英語）](https://mlflow.org/docs/latest/llms/index.html)
# MAGIC - [Databricks生成AIガイド（日本語）](https://docs.databricks.com/ja/generative-ai/index.html)
# MAGIC - [MLflow Tracing詳細（英語）](https://mlflow.org/docs/latest/llms/tracing/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🎉 お疲れ様でした！
# MAGIC
# MAGIC 生成AIアプリケーションのトレーシングと評価の基本を学びました。
# MAGIC
# MAGIC これらのテクニックを使って、高品質なLLMアプリケーションを構築してください！
