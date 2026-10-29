import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag_assistant.generation import NO_CONTEXT_ANSWER, generate_answer
from rag_assistant.retrieval import RetrievedChunk


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_no_context_message_without_calling_llm(self) -> None:
        with patch("rag_assistant.generation.create_openai_client") as create_client:
            answer = generate_answer("Вопрос?", [])

        self.assertEqual(answer, NO_CONTEXT_ANSWER)
        create_client.assert_not_called()

    def test_sends_context_and_question_to_chat_api(self) -> None:
        completion = Mock(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  Краткий ответ.  ")
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion
        context = [RetrievedChunk("Важный факт.", "guide.txt", 0.15)]

        with patch(
            "rag_assistant.generation.create_openai_client",
            return_value=client,
        ):
            answer = generate_answer("Что сказано?", context)

        self.assertEqual(answer, "Краткий ответ.")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("Отвечай только на основе предоставленного контекста", request["messages"][0]["content"])
        self.assertIn("[Источник: guide.txt]\nВажный факт.", request["messages"][1]["content"])
        self.assertIn("Что сказано?", request["messages"][1]["content"])

    def test_raises_when_chat_api_returns_empty_answer(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )

        with (
            patch("rag_assistant.generation.create_openai_client", return_value=client),
            self.assertRaises(RuntimeError),
        ):
            generate_answer(
                "Вопрос?",
                [RetrievedChunk("Факт.", "guide.txt", 0.1)],
            )

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            generate_answer("", [RetrievedChunk("Факт.", "guide.txt", 0.1)])


if __name__ == "__main__":
    unittest.main()
