"""
GRACE Config Tests
設定のテスト
"""

import pytest
import os
import tempfile
from pathlib import Path

from grace.config import (
    GraceConfig,
    LLMConfig,
    EmbeddingConfig,
    ConfidenceConfig,
    ConfigLoader,
    get_config,
    reload_config,
    reset_config,
)


class TestConfigModels:
    """設定モデルのテスト"""

    def test_llm_config_defaults(self):
        """LLMConfig のデフォルト値"""
        config = LLMConfig()

        assert config.provider == "gemini"
        assert config.model == "gemini-2.5-flash"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 30

    def test_embedding_config_defaults(self):
        """EmbeddingConfig のデフォルト値"""
        config = EmbeddingConfig()

        assert config.provider == "gemini"
        assert config.model == "gemini-embedding-001"
        assert config.dimensions == 3072

    def test_confidence_config_defaults(self):
        """ConfidenceConfig のデフォルト値"""
        config = ConfidenceConfig()

        assert config.weights.search_quality == 0.25
        assert config.weights.llm_self_eval == 0.25
        assert config.thresholds.silent == 0.9
        assert config.thresholds.confirm == 0.4

    def test_grace_config_defaults(self):
        """GraceConfig のデフォルト値"""
        config = GraceConfig()

        assert config.version == "1.0"
        assert config.llm.provider == "gemini"
        assert config.embedding.dimensions == 3072
        assert config.replan.max_replans == 3
        assert config.cost.daily_limit_usd == 10.0

    def test_custom_config(self):
        """カスタム設定"""
        config = GraceConfig(
            llm=LLMConfig(model="gemini-2.5-flash-preview"),
            cost={"daily_limit_usd": 5.0}
        )

        assert config.llm.model == "gemini-2.5-flash-preview"
        assert config.cost.daily_limit_usd == 5.0


class TestConfigLoader:
    """ConfigLoaderのテスト"""

    def setup_method(self):
        """各テスト前にリセット"""
        reset_config()

    def test_load_defaults_when_no_file(self):
        """設定ファイルがない場合はデフォルト値"""
        loader = ConfigLoader(config_path="nonexistent.yml")
        config = loader.load()

        assert isinstance(config, GraceConfig)
        assert config.llm.model == "gemini-2.5-flash"

    def test_load_from_yaml(self):
        """YAMLファイルから読み込み"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yml',
            delete=False
        ) as f:
            f.write("""
llm:
  model: "test-model"
  temperature: 0.5
cost:
  daily_limit_usd: 5.0
""")
            f.flush()

            try:
                loader = ConfigLoader(config_path=f.name)
                config = loader.load()

                assert config.llm.model == "test-model"
                assert config.llm.temperature == 0.5
                assert config.cost.daily_limit_usd == 5.0
            finally:
                os.unlink(f.name)

    def test_env_override(self):
        """環境変数による上書き"""
        os.environ["GRACE_LLM_MODEL"] = "env-model"
        os.environ["GRACE_COST_DAILY_LIMIT_USD"] = "3.0"

        try:
            loader = ConfigLoader(config_path="nonexistent.yml")
            config = loader.load()

            assert config.llm.model == "env-model"
            assert config.cost.daily_limit_usd == 3.0
        finally:
            del os.environ["GRACE_LLM_MODEL"]
            del os.environ["GRACE_COST_DAILY_LIMIT_USD"]

    def test_reload_config(self):
        """設定の再読み込み"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yml',
            delete=False
        ) as f:
            f.write("""
llm:
  model: "original-model"
""")
            f.flush()

            try:
                loader = ConfigLoader(config_path=f.name)
                config1 = loader.load()
                assert config1.llm.model == "original-model"

                # ファイルを更新
                with open(f.name, 'w') as f2:
                    f2.write("""
llm:
  model: "updated-model"
""")

                config2 = loader.reload()
                assert config2.llm.model == "updated-model"
            finally:
                os.unlink(f.name)


class TestGetConfig:
    """get_config関数のテスト"""

    def setup_method(self):
        """各テスト前にリセット"""
        reset_config()

    def test_singleton_behavior(self):
        """シングルトンの動作"""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reset_config(self):
        """設定リセット"""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        # 新しいインスタンスが作成される
        assert config1 is not config2


class TestConfigValidation:
    """設定値の検証テスト"""

    def test_confidence_weights_sum(self):
        """Confidence重みの合計は1.0"""
        config = GraceConfig()
        weights = config.confidence.weights

        total = (
            weights.search_quality +
            weights.source_agreement +
            weights.llm_self_eval +
            weights.tool_success +
            weights.query_coverage
        )

        assert abs(total - 1.0) < 0.01

    def test_thresholds_order(self):
        """閾値の順序"""
        config = GraceConfig()
        thresholds = config.confidence.thresholds

        assert thresholds.silent > thresholds.notify
        assert thresholds.notify > thresholds.confirm