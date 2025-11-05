# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow もくもく会 - 生成AIのトレーシング & 評価
# MAGIC
# MAGIC このノートブックでは、**MLflowを使った生成AIアプリケーションのトレーシングと評価**を学びます。
# MAGIC
# MAGIC ## 本日の流れ
# MAGIC 1. **環境セットアップ**: 必要なライブラリのインストール
# MAGIC 2. **Part 1**: Tracingの基本とAuto-tracing
# MAGIC 3. **Part 2**: GenAI Evaluationの実践（事前構築されたJudge）
# MAGIC 4. **Part 3**: モデルのロギングとデプロイ
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
# MAGIC - **Databricks Agents**: GenAI評価用
# MAGIC - **OpenAI**: MLflow OpenAIトレーシング用
# MAGIC - **Pandas**: データ処理

# COMMAND ----------

# MLflowとDatabricks SDKのインストール
%pip install mlflow databricks-sdk databricks-agents openai pandas plotly
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
# MAGIC
# MAGIC **span_type（スパンタイプ）**を設定することで、トレース画面に適切なアイコンが表示されます：
# MAGIC - `LLM`: LLM呼び出し（💬アイコン）
# MAGIC - `RETRIEVER`: ドキュメント検索（🔍アイコン）
# MAGIC - `CHAIN`: 複数ステップの処理（🔗アイコン）
# MAGIC - `TOOL`: ツール呼び出し（🔧アイコン）

# COMMAND ----------

import mlflow
import mlflow.openai

# Auto-tracingを有効化（OpenAI互換API用）
mlflow.openai.autolog()

# トレースされる関数
@mlflow.trace(span_type="LLM")
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

@mlflow.trace(name="retrieve_documents", span_type="RETRIEVER")
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

@mlflow.trace(name="generate_answer", span_type="LLM")
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

@mlflow.trace(name="rag_pipeline", span_type="CHAIN")
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
print("📊 右側の「Traces」タブから詳細を確認できます")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: GenAI Evaluationの実践
# MAGIC
# MAGIC LLMアプリケーションの品質を評価します。
# MAGIC
# MAGIC ## 評価の重要性
# MAGIC
# MAGIC LLMの出力は確率的なので、以下を評価する必要があります：
# MAGIC - **正確性（Correctness）**: 回答が正しいか
# MAGIC - **関連性（Relevance）**: 質問に関連しているか
# MAGIC - **安全性（Safety）**: 有害なコンテンツを含まないか
# MAGIC - **幻覚（Groundedness）**: 事実と異なる情報を生成していないか
# MAGIC
# MAGIC ## 事前構築されたJudge
# MAGIC
# MAGIC MLflowは以下の事前構築されたJudgeを提供しています：
# MAGIC - `RelevanceToQuery`: 質問への関連性
# MAGIC - `Correctness`: 正確性（グラウンドトゥルースとの比較）
# MAGIC - `Safety`: 安全性（有害コンテンツの検出）
# MAGIC - `RetrievalGroundedness`: 幻覚の検出
# MAGIC - `Guidelines`: ガイドライン準拠
# MAGIC
# MAGIC 📖 参考リンク：
# MAGIC - [事前構築されたJudges（日本語）](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/concepts/judges/pre-built-judges-scorers)
# MAGIC - [LLM Evaluation Guide（英語）](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価用のデータセットを準備

# COMMAND ----------

import pandas as pd

# 評価用のQAデータセット
# mlflow.genai.evaluate()では 'inputs' と 'expectations' カラムが必要
# inputs と expectations はどちらも辞書形式である必要がある
eval_data = pd.DataFrame({
    "inputs": [
        {"question": "MLflowとは何ですか？"},
        {"question": "MLflowのトラッキング機能はどのように動作しますか？"},
        {"question": "モデルレジストリとは何ですか？"},
        {"question": "MLflowを使ってモデルをデプロイする方法は？"},
        {"question": "MLflow Projectsとは何ですか？"}
    ],
    "expectations": [
        {"expected_response": "MLflowは機械学習ライフサイクル全体を管理するためのオープンソースプラットフォームです。"},
        {"expected_response": "MLflow Trackingを使用すると、MLコードを実行する際にパラメータ、メトリクス、アーティファクトをログできます。"},
        {"expected_response": "モデルレジストリは、モデルのバージョンとライフサイクルを管理するための集中型モデルストアです。"},
        {"expected_response": "MLflow Modelsは標準フォーマットを使用して、さまざまなプラットフォームにデプロイできます。"},
        {"expected_response": "MLflow Projectsは、再利用可能で再現可能な形式でMLコードをパッケージ化します。"}
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

# 予測を生成（テスト用）
eval_data["prediction"] = eval_data["inputs"].apply(lambda x: qa_model(x["question"]))

print("=== 予測結果 ===")
display(eval_data[["inputs", "prediction"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 事前構築されたJudgeを使った評価
# MAGIC
# MAGIC MLflowの事前構築されたJudge（評価者）を使います。

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Correctness, Safety, RetrievalGroundedness, Guidelines

# 評価用のpredict関数を作成
# predict_fnの引数名は、inputsカラムの辞書のキーと一致する必要がある
def predict_fn(question):
    """
    評価用のpredict関数
    question: 質問文（文字列）
    returns: 予測結果（文字列）
    """
    return qa_model(question)

# Guidelines Judge（ガイドライン準拠）の作成
guidelines = Guidelines(
    guidelines="""
    良い回答の基準:
    1. 簡潔で分かりやすい（50-300文字）
    2. 専門用語を適切に使用
    3. 具体的な説明を含む
    """
)

# 評価の実行
with mlflow.start_run(run_name="QA Model Evaluation with All Judges"):

    # すべての事前構築されたJudge（評価者）を定義
    judges = [
        RelevanceToQuery(),        # 質問への関連性
        Correctness(),             # 正確性（グラウンドトゥルースと比較）
        Safety(),                  # 安全性（有害コンテンツの検出）
        RetrievalGroundedness(),   # 幻覚の検出
        guidelines                 # ガイドライン準拠
    ]

    print("=== 使用するJudges ===")
    print("1. RelevanceToQuery: 質問への関連性")
    print("2. Correctness: 正確性（グラウンドトゥルースとの比較）")
    print("3. Safety: 安全性（有害コンテンツの検出）")
    print("4. RetrievalGroundedness: 幻覚の検出")
    print("5. Guidelines: カスタムガイドライン準拠")
    print()

    # mlflow.genai.evaluateで評価
    eval_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_fn,
        scorers=judges
    )

    print("=== 評価結果 ===")
    print(f"\n評価スコア:")
    for metric_name, metric_value in eval_results.metrics.items():
        print(f"  {metric_name}: {metric_value:.3f}")

    print("\n✅ すべてのJudgeによる評価が完了しました！")
    print("📊 詳細な評価結果は右側の「Experiments」タブから確認できます")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: モデルのロギングとデプロイ
# MAGIC
# MAGIC 評価が完了したら、モデルをLoggedModelとしてロギングし、サービングエンドポイントにデプロイします。
# MAGIC
# MAGIC ## デプロイの流れ
# MAGIC
# MAGIC 1. **LoggedModelとしてロギング**: RAGパイプラインをMLflowモデルとして保存
# MAGIC 2. **Unity Catalogに登録**: モデルをUnity Catalogのモデルレジストリに登録
# MAGIC 3. **サービングエンドポイントにデプロイ**: `databricks.agents.deploy()`でデプロイ
# MAGIC 4. **エンドポイントのテスト**: デプロイされたエンドポイントを呼び出してテスト
# MAGIC
# MAGIC 📖 参考リンク：[Agent評価とデプロイ（日本語）](https://docs.databricks.com/ja/generative-ai/deploy-agent.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LoggedModelとしてRAGパイプラインをロギング

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

# RAGパイプラインをMLflowモデルとしてラップ
class RAGModel(mlflow.pyfunc.PythonModel):
    """
    RAGパイプラインをMLflowモデルとしてラップ
    """
    def predict(self, context, model_input):
        """
        予測関数
        model_input: {"question": "質問文"} の形式
        """
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        else:
            questions = [model_input["question"]]

        results = []
        for question in questions:
            result = rag_pipeline(question)
            results.append(result["answer"])

        return results

# モデルをロギング
with mlflow.start_run(run_name="RAG Model Logging"):
    # サンプル入力でシグネチャを推論
    sample_input = pd.DataFrame({"question": ["MLflowとは何ですか？"]})
    sample_output = rag_pipeline("MLflowとは何ですか？")["answer"]
    signature = infer_signature(sample_input, [sample_output])

    # モデルをロギング
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        signature=signature,
        input_example=sample_input
    )

    # モデルURIを取得
    model_uri = mlflow.get_artifact_uri("rag_model")
    print(f"✅ モデルをロギングしました: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalogに登録

# COMMAND ----------

# 最新のランからモデルを取得
latest_run = mlflow.search_runs(order_by=["start_time desc"]).iloc[0]
model_uri = f"runs:/{latest_run.run_id}/rag_model"

# Unity Catalogに登録
uc_model_name = "workspace.default.rag_qa_model"

try:
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=uc_model_name
    )
    print(f"✅ Unity Catalogに登録しました:")
    print(f"   モデル名: {uc_model_name}")
    print(f"   バージョン: {model_version.version}")
except Exception as e:
    print(f"⚠️ Unity Catalog登録エラー: {e}")
    print("   このステップはオプションです。次に進んでください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## サービングエンドポイントにデプロイ

# COMMAND ----------

from databricks import agents

# エンドポイント名
endpoint_name = "rag-qa-endpoint"

print("=== サービングエンドポイントへのデプロイ ===\n")

try:
    # エンドポイントにデプロイ
    deployment = agents.deploy(
        model_name=uc_model_name,
        model_version=model_version.version if 'model_version' in locals() else 1,
        endpoint_name=endpoint_name
    )

    print(f"✅ デプロイ開始:")
    print(f"   エンドポイント名: {endpoint_name}")
    print(f"   モデル: {uc_model_name}")
    print("\n📝 デプロイには数分かかります")
    print("📊 Databricks UIの「サービングエンドポイント」から進捗を確認できます")

except Exception as e:
    print(f"⚠️ デプロイエラー: {e}")
    print("\n代替案: 手動でデプロイする場合:")
    print("1. Databricks UIの「サービングエンドポイント」に移動")
    print(f"2. モデル '{uc_model_name}' を選択してエンドポイントを作成")

# COMMAND ----------

# MAGIC %md
# MAGIC ## デプロイされたエンドポイントのテスト

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("=== エンドポイントのテスト ===\n")

try:
    # エンドポイントを呼び出し
    response = w.serving_endpoints.query(
        name=endpoint_name,
        inputs={"question": "MLflowとは何ですか？"}
    )

    print("✅ エンドポイント呼び出し成功:")
    print(f"   質問: MLflowとは何ですか？")
    print(f"   回答: {response}\n")

    print("✅ エンドポイントが正常に動作しています")
    print("📊 本番トラフィックのトレースが記録されます")
    print("📊 次のPart 4でこれらのトレースに自動評価を設定します")

except Exception as e:
    print(f"⚠️ エンドポイント呼び出しエラー: {e}")
    print("\n考えられる原因:")
    print("- エンドポイントのデプロイが完了していない（数分待ってから再試行）")
    print("- エンドポイント名が間違っている")
    print("- 権限が不足している")

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: 本番環境モニタリング
# MAGIC
# MAGIC **前提条件**: Part 3でサービングエンドポイントにデプロイ済み
# MAGIC
# MAGIC デプロイされたエンドポイントで発生する本番トラフィックのトレースに対して、**自動的に評価を実行**します。
# MAGIC
# MAGIC ## 本番環境モニタリングとは
# MAGIC
# MAGIC サービングエンドポイントが受け取るリクエストは自動的にトレースされます。これらのトレースに対して、登録されたScorerが**バックグラウンドで自動評価**を実行します。
# MAGIC
# MAGIC ### 主要な機能
# MAGIC
# MAGIC 1. **Scorerの登録とアクティブ化**
# MAGIC    - `.register()`: Scorerを登録
# MAGIC    - `.start()`: サンプリング設定でアクティブ化
# MAGIC
# MAGIC 2. **サンプリング設定**
# MAGIC    - `sample_rate=1.0`: 全トレースを評価（重要な指標）
# MAGIC    - `sample_rate=0.1`: 10%のみ評価（コスト削減）
# MAGIC
# MAGIC 3. **バックフィル**
# MAGIC    - 過去のトレースに遡及的に評価を適用
# MAGIC
# MAGIC 4. **Scorerのライフサイクル**
# MAGIC    - 未登録 → 登録済み → アクティブ ↔ 停止 → 削除済み
# MAGIC
# MAGIC 📖 参考リンク：[本番環境モニタリング（日本語）](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/production-monitoring)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorerの登録とアクティブ化（コード例）
# MAGIC
# MAGIC ⚠️ **注意**: このセクションはデモ用のコード例です。実際の本番環境では有効化されません。

# COMMAND ----------

from mlflow.genai.scorers import Safety, RelevanceToQuery, ScorerSamplingConfig

print("=== 本番環境モニタリングの設定方法 ===\n")

# ステップ1: Scorerを登録
print("【ステップ1】Scorerの登録")
print("以下のコードでScorerを登録します：\n")
print("safety_scorer = Safety().register(name='production_safety')")
print("relevance_scorer = RelevanceToQuery().register(name='production_relevance')\n")

# ステップ2: Scorerをアクティブ化
print("【ステップ2】Scorerのアクティブ化とサンプリング設定")
print("サンプリング設定でアクティブ化します：\n")
print("# 安全性チェックは全トレースで実行（重要）")
print("safety_scorer = safety_scorer.start(")
print("    sampling_config=ScorerSamplingConfig(sample_rate=1.0)")
print(")\n")
print("# 関連性チェックは20%のトレースのみ（コスト削減）")
print("relevance_scorer = relevance_scorer.start(")
print("    sampling_config=ScorerSamplingConfig(sample_rate=0.2)")
print(")\n")

# ステップ3: Scorerの状態管理
print("【ステップ3】Scorerの状態管理")
print("Scorerは以下の状態を持ちます：")
print("- 未登録 → .register() → 登録済み")
print("- 登録済み → .start() → アクティブ")
print("- アクティブ → .stop() → 停止")
print("- 停止 → .start() → アクティブ（再開）")
print("- 任意の状態 → .delete() → 削除済み\n")

print("✅ 本番環境モニタリングの設定方法を確認しました")
print("📊 実際の環境では、新しいトレースが自動的に評価されます")

# COMMAND ----------

# MAGIC %md
# MAGIC ## バックフィル機能（過去のトレースを評価）
# MAGIC
# MAGIC 過去のトレースに遡及的に評価を適用できます。

# COMMAND ----------

print("=== バックフィル機能 ===\n")

print("過去のトレースに評価を適用するコード例：\n")
print("from databricks.agents.scorers import backfill_scorers")
print("")
print("# 特定のScorerでバックフィル")
print("job_id = backfill_scorers(")
print("    scorers=['production_safety', 'production_relevance']")
print(")\n")

print("📝 バックフィルは非同期で実行され、完了まで15-20分程度かかります")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 本番環境モニタリングのベストプラクティス

# COMMAND ----------

print("=== 本番環境モニタリングのベストプラクティス ===\n")

practices = [
    ("1. 適切なサンプリングレート", [
        "• 安全性・セキュリティチェック: sample_rate=1.0（全トレース）",
        "• コストの高い評価: sample_rate=0.05～0.2（5-20%）",
        "• 一般的な品質指標: sample_rate=0.5（50%）"
    ]),
    ("2. Scorer数の管理", [
        "• エクスペリメントあたり最大20個のScorerまで",
        "• 本当に必要な評価に絞る"
    ]),
    ("3. 処理時間の考慮", [
        "• 初回スコアリング: 15-20分程度かかる",
        "• その後は新しいトレースに自動適用される"
    ]),
    ("4. カスタムScorerの注意点", [
        "• 外部依存はスコアラー関数内でインポートする",
        "• 状態を保持しないステートレスな実装にする"
    ])
]

for title, items in practices:
    print(f"【{title}】")
    for item in items:
        print(item)
    print()

print("\n✅ 本番環境モニタリングの設定完了")
print("📖 詳細は公式ドキュメントを参照してください")

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
# MAGIC - すべての事前構築されたJudgeで品質測定
# MAGIC - RelevanceToQuery, Correctness, Safety, RetrievalGroundedness, Guidelines
# MAGIC - mlflow.genai.evaluate()で一括評価
# MAGIC
# MAGIC ### ✅ モデルのロギングとデプロイ
# MAGIC - LoggedModelとしてRAGパイプラインを保存
# MAGIC - Unity Catalogに登録
# MAGIC - agents.deployでサービングエンドポイントにデプロイ
# MAGIC
# MAGIC ### ✅ 本番モニタリング
# MAGIC - Scorerの登録とアクティブ化で自動評価
# MAGIC - サンプリング設定でコスト最適化
# MAGIC - バックフィル機能で過去のトレースも評価
# MAGIC
# MAGIC ## 次のステップ
# MAGIC
# MAGIC - **実際のデプロイ**: agents.deployで本番環境にデプロイ
# MAGIC - **より高度な評価**: 複数のJudgeを組み合わせた評価
# MAGIC - **A/Bテスト**: 複数のモデルバージョンを比較
# MAGIC - **モニタリングの最適化**: サンプリングレートの調整とコスト最適化
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
