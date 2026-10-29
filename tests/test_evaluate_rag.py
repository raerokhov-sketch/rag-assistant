import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.evaluate_rag import prepare_dataset
from rag_assistant.retrieval import RetrievedChunk


class PrepareRagasDatasetTests(unittest.TestCase):
    def test_builds_dataset_from_assistant_answers_and_retrieved_context(self) -> None:
        question = "Что есть в документе?"
        context = [RetrievedChunk("Нужный факт.", "guide.txt", 0.1)]
        dataset = SimpleNamespace()

        with (
            patch(
                "evaluation.evaluate_rag.answer_question_with_context",
                return_value=("Ответ ассистента.", context),
            ) as answer_question,
            patch(
                "datasets.Dataset.from_dict",
                return_value=dataset,
            ) as from_dict,
        ):
            result = prepare_dataset([question])

        self.assertIs(result, dataset)
        answer_question.assert_called_once_with(question)
        self.assertEqual(
            from_dict.call_args.args[0],
            {
                "question": [question],
                "answer": ["Ответ ассистента."],
                "contexts": [["Нужный факт."]],
                "ground_truth": [""],
            },
        )

    def test_builds_and_runs_ragas_metrics_with_configured_clients(self) -> None:
        dataset = object()
        evaluation_result = SimpleNamespace(
            scores=[
                {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.7,
                    "context_precision": 0.9,
                }
            ]
        )

        with (
            patch("evaluation.evaluate_rag.prepare_dataset", return_value=dataset),
            patch("evaluation.evaluate_rag.require_api_key", return_value="test-key"),
            patch("ragas.evaluate", return_value=evaluation_result) as evaluate,
        ):
            from evaluation.evaluate_rag import evaluate_rag_system

            evaluate_rag_system()

        self.assertIs(evaluate.call_args.kwargs["dataset"], dataset)
        self.assertEqual(
            [metric.name for metric in evaluate.call_args.kwargs["metrics"]],
            ["faithfulness", "answer_relevancy", "context_precision"],
        )


if __name__ == "__main__":
    unittest.main()
