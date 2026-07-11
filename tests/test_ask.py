from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from rag.ask import NO_RELEVANT_CONTEXT_MESSAGE, _answer_question, main


class AnswerQuestionTests(unittest.TestCase):
    def test_rejection_does_not_call_model(self) -> None:
        chat = Mock()
        output = io.StringIO()

        with patch("rag.ask.retrieve", return_value=[]), patch("rag.ask.answer") as answer:
            with redirect_stdout(output):
                _answer_question(chat, "unrelated question", top_k=5, layer=None, stream=True)

        answer.assert_not_called()
        self.assertIn(NO_RELEVANT_CONTEXT_MESSAGE, output.getvalue())

    def test_one_shot_rejection_does_not_load_model(self) -> None:
        output = io.StringIO()

        with patch("sys.argv", ["career-rag-ask", "unrelated question"]):
            with patch("rag.ask.retrieve", return_value=[]):
                with patch("rag.ask.load_model") as load_model:
                    with redirect_stdout(output):
                        main()

        load_model.assert_not_called()
        self.assertIn(NO_RELEVANT_CONTEXT_MESSAGE, output.getvalue())


if __name__ == "__main__":
    unittest.main()
