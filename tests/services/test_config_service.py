import os
import pytest
from unittest.mock import patch, mock_open
from services.config_service import ConfigManager

# Reset singleton before each test
@pytest.fixture(autouse=True)
def reset_singleton():
    ConfigManager._instance = None
    yield
    ConfigManager._instance = None

class TestConfigManager:

    def test_singleton(self):
        cm1 = ConfigManager()
        cm2 = ConfigManager()
        assert cm1 is cm2

    def test_load_default(self):
        with patch("services.config_service.Path.exists", return_value=False):
            cm = ConfigManager()
            # Should have defaults
            assert cm.get("api.timeout") == 30
            assert cm.get("models.default") == "gemini-2.0-flash"

    def test_load_yaml(self):
        yaml_content = """
api:
  timeout: 60
models:
  default: "gpt-4"
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)), \
             patch("services.config_service.Path.exists", return_value=True):
            
            cm = ConfigManager()
            assert cm.get("api.timeout") == 60
            assert cm.get("models.default") == "gpt-4"

    def test_env_override(self):
        with patch("services.config_service.Path.exists", return_value=False), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            
            cm = ConfigManager()
            assert cm.get("api.openai_api_key") == "env_key"

    def test_get_set(self):
        with patch("services.config_service.Path.exists", return_value=False):
            cm = ConfigManager()
            
            cm.set("new.key", "value")
            assert cm.get("new.key") == "value"
            
            # Cache check
            cm.set("new.key", "updated")
            assert cm.get("new.key") == "updated"

    def test_get_nested_missing(self):
         with patch("services.config_service.Path.exists", return_value=False):
            cm = ConfigManager()
            assert cm.get("non.existent.key", "default_val") == "default_val"

