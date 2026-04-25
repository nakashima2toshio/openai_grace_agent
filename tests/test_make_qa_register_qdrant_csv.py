
import unittest
import pandas as pd
import os
import shutil
from unittest.mock import MagicMock, patch

class TestMakeQaRegisterQdrantCSV(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "tests/temp_csv_test"
        os.makedirs(self.test_dir, exist_ok=True)
        self.csv_path = os.path.join(self.test_dir, "test.csv")
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ui_csv_creation_success(self):
        """正常系: UI用CSV作成が成功するか"""
        df = pd.DataFrame({'question': ['q'], 'answer': ['a'], 'other': ['o']})
        output_path = os.path.join(self.test_dir, "ui_output.csv")
        
        # ロジック再現
        try:
            df[['question', 'answer']].to_csv(output_path, index=False, encoding='utf-8')
            self.assertTrue(os.path.exists(output_path))
        except Exception as e:
            self.fail(f"CSV作成失敗: {e}")

    def test_ui_csv_creation_missing_columns(self):
        """異常系: 必須カラム欠損時の挙動"""
        df = pd.DataFrame({'other': ['o']}) # question, answerなし
        output_path = os.path.join(self.test_dir, "ui_output_fail.csv")
        
        # ロジック再現 (KeyErrorが発生するはず)
        with self.assertRaises(KeyError):
            df[['question', 'answer']].to_csv(output_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    unittest.main()
