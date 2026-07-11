from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from rag.retrieve import retrieve


def query_result(scores: list[float]) -> dict:
    return {
        "documents": [[f"document-{index}" for index in range(len(scores))]],
        "metadatas": [[{"full_path": f"path-{index}"} for index in range(len(scores))]],
        "distances": [[1 - score for score in scores]],
    }


class RetrieveTests(unittest.TestCase):
    def test_rejects_all_hits_below_absolute_floor(self) -> None:
        collection = Mock()
        collection.query.return_value = query_result([0.29, 0.20, 0.10])

        with patch("rag.retrieve.get_collection", return_value=collection):
            hits = retrieve("unrelated question", min_score=0.30)

        self.assertEqual(hits, [])

    def test_keeps_strong_hits_close_to_best_score(self) -> None:
        collection = Mock()
        collection.query.return_value = query_result([0.80, 0.70, 0.60, 0.20])

        with patch("rag.retrieve.get_collection", return_value=collection):
            hits = retrieve("career question", top_k=5, keep_ratio=0.85, min_score=0.30)

        self.assertEqual([hit["score"] for hit in hits], [0.80, 0.70])

    def test_empty_query_is_rejected_without_opening_collection(self) -> None:
        with patch("rag.retrieve.get_collection") as get_collection:
            hits = retrieve("   ")

        self.assertEqual(hits, [])
        get_collection.assert_not_called()

    def test_invalid_retrieval_parameters_are_rejected(self) -> None:
        invalid_arguments = [
            {"top_k": 0},
            {"keep_ratio": 0},
            {"keep_ratio": 1.1},
            {"min_score": -1.1},
            {"min_score": 1.1},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    retrieve("career question", **arguments)


if __name__ == "__main__":
    unittest.main()
