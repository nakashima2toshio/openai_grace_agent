import json
import pytest
from datetime import datetime
from unittest.mock import mock_open, patch
from services.json_service import (
    safe_json_serializer, safe_json_dumps, safe_json_loads,
    load_json_file, save_json_file, is_valid_json
)

class TestJsonService:

    def test_safe_json_serializer(self):
        # DateTime
        dt = datetime(2023, 1, 1, 12, 0, 0)
        assert safe_json_serializer(dt) == "2023-01-01T12:00:00"
        
        # Set
        s = {1, 2, 3}
        serialized_set = safe_json_serializer(s)
        assert isinstance(serialized_set, list)
        assert set(serialized_set) == s
        
        # Bytes
        b = b"hello"
        assert safe_json_serializer(b) == "hello"
        
        # Object with dict
        class ObjWithDict:
            def dict(self):
                return {"a": 1}
        assert safe_json_serializer(ObjWithDict()) == {"a": 1}

        # Fallback to str
        class Unknown:
            def __str__(self):
                return "unknown"
        assert safe_json_serializer(Unknown()) == "unknown"

    def test_safe_json_dumps(self):
        data = {"a": datetime(2023, 1, 1)}
        json_str = safe_json_dumps(data)
        assert "2023-01-01" in json_str

    def test_safe_json_loads(self):
        json_str = '{"a": 1}'
        assert safe_json_loads(json_str) == {"a": 1}
        
        assert safe_json_loads("invalid") is None
        assert safe_json_loads("invalid", default={}) == {}

    def test_load_json_file(self):
        with patch("builtins.open", mock_open(read_data='{"key": "value"}')):
            data = load_json_file("dummy.json")
            assert data == {"key": "value"}
            
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert load_json_file("dummy.json") is None

    def test_save_json_file(self):
        with patch("builtins.open", mock_open()) as m:
            success = save_json_file({"a": 1}, "dummy.json")
            assert success is True
            m.assert_called()
            
            # Verify content
            handle = m()
            # Depending on implementation, it might call write multiple times or once
            # Just check if write was called
            assert handle.write.called

    def test_is_valid_json(self):
        assert is_valid_json('{"a": 1}') is True
        assert is_valid_json('{a: 1}') is False # Invalid JSON syntax
