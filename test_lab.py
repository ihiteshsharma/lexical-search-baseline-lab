import tempfile
import unittest
from pathlib import Path

from search_lab import demo, evaluate, initialize, rebuild, remove_from_index, search


class RetrievalBaselineIntegrationTest(unittest.TestCase):
    def test_demo_states_control_and_representation_limit(self):
        report = demo()
        self.assertEqual(report["verdict"], "passed")
        self.assertTrue(all(report["checks"].values()))
        self.assertIn("reproducible control", report["conclusion"])
        self.assertIn("synonymy", report["conclusion"])

    def test_index_loss_fails_the_gate_and_rebuild_restores_exact_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "search.db"
            initialize(db)
            rebuild(db)

            baseline = evaluate(db)
            self.assertEqual(baseline["source_count"], 20)
            self.assertEqual(baseline["index_count"], 20)
            self.assertEqual(baseline["recall_at_3"], 1.0)
            self.assertEqual(baseline["ndcg_at_3"], 0.958498)
            self.assertTrue(baseline["gate_passed"])

            remove_from_index(db, "DOC-007")
            broken = evaluate(db)
            self.assertFalse(broken["gate_passed"])
            self.assertEqual(broken["index_count"], 19)

            rebuild(db)
            self.assertEqual(evaluate(db), baseline)
            self.assertEqual(search(db, "automobile"), [])


if __name__ == "__main__":
    unittest.main()
