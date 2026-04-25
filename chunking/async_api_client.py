# async_api_client.py
"""
非同期APIクライアント
- asyncio.to_thread() で同期APIをラップ
- Semaphore で並列数制御（固定）
- リトライロジック（3回、指数バックオフ）
- 不完全JSONの検出とリトライ
"""

import asyncio
import json
import logging
from typing import Type, Optional

from pydantic import BaseModel
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class AsyncAPIClient:
    """
    非同期APIクライアント
    - asyncio.to_thread() で同期APIをラップ
    - Semaphore で並列数制御（固定）
    - リトライロジック（3回、指数バックオフ）
    - 不完全JSONの検出とリトライ
    """

    def __init__(
        self,
        api_key: str,
        max_workers: int = 8,
        max_retries: int = 3,
        max_output_tokens: int = 8192
    ):
        """
        Args:
            api_key: Google API Key
            max_workers: 並列数（デフォルト: 8、固定）
            max_retries: リトライ回数（デフォルト: 3）
            max_output_tokens: 出力トークン制限（デフォルト: 4096）
        """
        self.client = genai.Client(api_key=api_key)
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._total_requests = 0
        self._failed_requests = 0
        self._truncated_responses = 0

    def _is_valid_json(self, text: str) -> bool:
        """JSONが完全かどうかチェック"""
        if not text:
            return False
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _is_truncated_response(self, response) -> bool:
        """レスポンスが切断されたかチェック"""
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                # finish_reason が None の場合は正常とみなす
                if finish_reason is None:
                    return False
                
                # 文字列の場合
                if isinstance(finish_reason, str):
                    return finish_reason.upper() not in ['STOP', 'END']
                
                # Enum の場合（値が 1 = STOP）
                if hasattr(finish_reason, 'value'):
                    return finish_reason.value not in [1, 'STOP']
                
                # int の場合
                if isinstance(finish_reason, int):
                    return finish_reason != 1
                    
        except Exception as e:
            logger.debug(f"Error checking finish_reason: {e}")
        return False

    async def generate_content(
        self,
        model: str,
        contents: str,
        response_schema: Type[BaseModel],
        task_id: Optional[str] = None
    ) -> Optional[str]:
        """
        セマフォで並列数を制御しながらAPI呼び出し
        失敗時は指数バックオフでリトライ
        Args:
            model: Geminiモデル名
            contents: 入力テキスト
            response_schema: レスポンスのPydanticスキーマ
            task_id: タスク識別子（ログ用）
        Returns:
            レスポンステキスト、または失敗時はNone
        """
        async with self.semaphore:
            return await self._execute_with_retry(
                model, contents, response_schema, task_id
            )

    async def _execute_with_retry(
        self,
        model: str,
        contents: str,
        response_schema: Type[BaseModel],
        task_id: Optional[str]
    ) -> Optional[str]:
        """リトライロジック（不完全JSON対策含む）"""

        for attempt in range(self.max_retries):
            try:
                self._total_requests += 1

                # asyncio.to_thread で同期APIを非同期実行
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )

                # レスポンス切断チェック
                if self._is_truncated_response(response):
                    self._truncated_responses += 1
                    finish_reason = "unknown"
                    if hasattr(response, 'candidates') and response.candidates:
                        finish_reason = getattr(response.candidates[0], 'finish_reason', 'unknown')
                    raise ValueError(f"Response truncated (finish_reason: {finish_reason})")

                # JSON完全性チェック
                if response.text and not self._is_valid_json(response.text):
                    self._truncated_responses += 1
                    preview = response.text[-100:] if len(response.text) > 100 else response.text
                    raise ValueError(
                        f"Incomplete JSON detected. "
                        f"Length: {len(response.text)}, "
                        f"End preview: ...{preview}"
                    )

                return response.text

            except ValueError as e:
                # 不完全レスポンス → リトライ
                wait_time = 2 ** attempt
                logger.warning(
                    f"[{task_id}] {e}. "
                    f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)

            except Exception as e:
                error_str = str(e).lower()

                # レート制限エラーの判定
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(
                        f"[{task_id}] Rate limit hit. "
                        f"Waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                else:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"[{task_id}] Error: {e}. "
                        f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                    )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)

        # 全リトライ失敗
        self._failed_requests += 1
        logger.error(f"[{task_id}] Failed after {self.max_retries} retries. Using fallback.")
        return None

    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "truncated_responses": self._truncated_responses,
            "success_rate": (
                (self._total_requests - self._failed_requests) / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "concurrency": self.max_workers
        }

    def reset_stats(self):
        """統計情報をリセット"""
        self._total_requests = 0
        self._failed_requests = 0
        self._truncated_responses = 0
