# MLflow もくもく会

MLflowを使った機械学習・生成AIのトレーニング教材です。Databricks環境で動作確認済みです。

## 📚 ノートブック一覧

### Notebook1: MLflowチュートリアル

基本的なMLflowの使い方を学ぶチュートリアルノートブックです。

**学べる内容:**
- MLflowの基本用語（Experiment、Run、Metrics、Parameters、Model）
- scikit-learnを使ったモデルトレーニング
- パラメータとメトリクスのロギング
- モデルの保存とシグネチャ
- 実験結果の可視化
- Unity Catalogモデルレジストリへの登録

**使用データセット:** Diabetes dataset (scikit-learn)
**使用モデル:** Linear Regression

### Notebook2: 生成AIのトレーシング & 評価

MLflowを使った生成AIアプリケーションのトレーシングと評価を学ぶノートブックです。

**学べる内容:**

#### Part 1: Tracingの基本とAuto-tracing
- `@mlflow.trace`デコレータによる自動トレーシング
- `span_type`パラメータでのアイコン表示（LLM、RETRIEVER、CHAIN）
- Databricks Foundation Model API (FMAPI) の使用
- RAGパイプラインの複数ステップ可視化
- デバッグとパフォーマンス最適化

#### Part 2: GenAI Evaluationの実践
- 5つの事前構築されたJudgeによる品質評価
  - **RelevanceToQuery**: 質問への関連性
  - **Correctness**: 正確性（グラウンドトゥルースとの比較）
  - **Safety**: 安全性（有害コンテンツの検出）
  - **RetrievalGroundedness**: 幻覚の検出
  - **Guidelines**: カスタムガイドライン準拠
- `mlflow.genai.evaluate()`による一括評価
- 評価データセットの作成

#### Part 3 & 4: モデルのロギング・デプロイ・本番モニタリング

Part 3（モデルのロギングとデプロイ）とPart 4（本番環境モニタリング）については、概要説明のみを含んでいます。詳細な実装は以下の記事を参照してください：

📖 **[MLflow3とDatabricksで実現するLLMops](https://qiita.com/taka_yayoi/items/2fd4c9fef0ffe8377f48)**

この記事では以下の内容を実装付きで解説しています：
- LoggedModelの実装方法
- Unity Catalogへの登録
- agents.deployを使ったデプロイ
- Scorerの登録とアクティブ化
- サンプリング設定とバックフィル
- 本番環境での継続的モニタリング

## 🚀 はじめ方

### 前提条件

- Databricksワークスペースへのアクセス
- Databricks Free Edition (Serverless Compute) で動作確認済み
- 必要なライブラリは各ノートブック内でインストール

### 使い方

1. このリポジトリをDatabricksワークスペースにクローン
2. Notebook1から順に実行して基礎を学習
3. Notebook2でLLMアプリケーションのトレーシングと評価を実践
4. Part 3以降の実装はQiita記事を参照して実際に試してみる

## 📖 参考リンク

### 公式ドキュメント
- [MLflow公式ドキュメント（英語）](https://mlflow.org/docs/latest/index.html)
- [MLflow LLMsガイド（英語）](https://mlflow.org/docs/latest/llms/index.html)
- [Databricks MLflowガイド（日本語）](https://docs.databricks.com/ja/mlflow/index.html)
- [Databricks生成AIガイド（日本語）](https://docs.databricks.com/ja/generative-ai/index.html)

### 関連記事
- [MLflow3とDatabricksで実現するLLMops（Qiita）](https://qiita.com/taka_yayoi/items/2fd4c9fef0ffe8377f48)
- [Databricks ML Getting Started](https://docs.databricks.com/aws/ja/getting-started/ml-get-started)

## 💡 よくある質問

**Q: どのノートブックから始めればいい？**

A: Notebook1から始めてください。MLflowの基本概念を理解してから、Notebook2で生成AI特有の機能を学ぶとスムーズです。

**Q: Databricks以外の環境で動作しますか？**

A: Notebook1はローカル環境でも動作しますが、Notebook2はDatabricks Foundation Model APIを使用しているため、Databricks環境が必要です。

**Q: トレーシングはコストがかかりますか？**

A: トレーシング自体のオーバーヘッドは最小限です。主なコストはLLM呼び出しによるものです。

**Q: 本番環境でトレーシングを使うべき？**

A: サンプリング設定を使えば、コストを抑えながら本番環境でも利用できます。詳細はQiita記事を参照してください。

## 🤝 コントリビューション

このリポジトリはもくもく会の教材として作成されています。改善提案やバグ報告は Issue でお知らせください。

## 📝 ライセンス

このプロジェクトは教育目的で作成されています。

---

**Happy Learning with MLflow! 🎉**
