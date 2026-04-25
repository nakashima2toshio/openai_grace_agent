import os
import sys
import logging
from dotenv import load_dotenv

# プロジェクトルートをパスに追加してインポート可能にする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.agent_service import ReActAgent
from config import AgentConfig

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_paris_income_question():
    """
    パリ市の平均世帯所得に関する質問を行い、エージェントの挙動を検証する。
    """
    # 1. 環境変数のロード
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping test: GEMINI_API_KEY not found.")
        return

    # 2. エージェントの初期化
    # wikipedia_ja コレクションのみを選択
    target_collection = "wikipedia_ja"
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
    
    print(f"\n--- Initializing Agent with collection: [{target_collection}] ---")
    agent = ReActAgent(selected_collections=[target_collection], model_name=model_name)

    # 3. 質問の定義
    question = "パリ市の平均世帯所得は、フランス全体の平均と比べてどうですか？多いですか？また、日本と比較するとどうですか？"
    print(f"\n--- User Question: {question} ---")

    # 4. エージェントの実行と検証
    print("\n--- Agent Execution Start ---")
    
    tool_called = False
    correct_collection_used = False
    final_answer_received = False

    # ジェネレータからイベントを取得
    for event in agent.execute_turn(question):
        event_type = event.get("type")
        content = event.get("content")

        if event_type == "log":
            # 思考プロセスのログ出力
            # print(f"[Log] {content}")
            pass

        elif event_type == "tool_call":
            tool_name = event.get("name")
            tool_args = event.get("args")
            print(f"\n[Tool Call] {tool_name} Args: {tool_args}")
            
            if tool_name == "search_rag_knowledge_base":
                tool_called = True
                # コレクションの指定を確認
                used_collection = tool_args.get("collection_name")
                if used_collection == target_collection:
                    correct_collection_used = True
                    print("  -> CORRECT: Target collection used.")
                else:
                    print(f"  -> WRONG: Expected {target_collection}, got {used_collection}")

        elif event_type == "tool_result":
            # ツール実行結果の表示（長い場合は切り詰め）
            result_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"[Tool Result] {result_preview}")

        elif event_type == "final_answer":
            print(f"\n[Final Answer]\n{content}")
            final_answer_received = True

    # 5. 結果の検証
    print("\n--- Verification Results ---")
    
    if tool_called:
        print("✅ Tool 'search_rag_knowledge_base' was called.")
    else:
        print("❌ Tool 'search_rag_knowledge_base' was NOT called.")

    if correct_collection_used:
        print(f"✅ Correct collection '{target_collection}' was used.")
    else:
        print(f"❌ Correct collection '{target_collection}' was NOT used.")

    if final_answer_received:
        print("✅ Final answer was received.")
    else:
        print("❌ Final answer was NOT received.")

    # 最終的な成功判定
    if tool_called and correct_collection_used and final_answer_received:
        print("\n🎉 TEST PASSED: Agent behaved as expected.")
    else:
        print("\n💥 TEST FAILED: Agent did not behave as expected.")

if __name__ == "__main__":
    test_paris_income_question()
