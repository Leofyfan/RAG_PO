import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from rag_po.attack.random_inject import build_random_poisoned_corpus
from rag_po.attack.semantic_inject import build_semantic_poisoned_corpus
from rag_po.data_prep.parse_pheme import parse_pheme_archive
from rag_po.defense.consistency import consistency_filter
from rag_po.defense.outlier_detect import semantic_outlier_filter
from rag_po.defense.social_rerank import social_credibility_score, social_rerank
from rag_po.evaluation.metrics import retrieval_metrics
from rag_po.models import TweetDocument
from rag_po.rag.vectordb import InMemoryVectorStore


def make_doc(doc_id, event="event", label="non-rumour", text="text", embedding=None, **meta):
    user = {
        "followers_count": meta.pop("followers_count", 10),
        "friends_count": meta.pop("friends_count", 10),
        "verified": meta.pop("verified", False),
        "statuses_count": meta.pop("statuses_count", 100),
        "listed_count": meta.pop("listed_count", 0),
        "created_at": meta.pop("created_at", "Mon Jan 01 00:00:00 +0000 2010"),
        "default_profile": meta.pop("default_profile", False),
    }
    return TweetDocument(
        doc_id=doc_id,
        event=event,
        label=label,
        text=text,
        created_at="Fri Jan 09 12:56:25 +0000 2015",
        retweet_count=meta.pop("retweet_count", 0),
        favorite_count=meta.pop("favorite_count", 0),
        user=user,
        thread_id=meta.pop("thread_id", doc_id),
        reaction_count=meta.pop("reaction_count", 0),
        deny_count=meta.pop("deny_count", 0),
        embedding=embedding,
        attack=meta.pop("attack", None),
    )


class ParsePhemeTests(unittest.TestCase):
    def test_parse_archive_extracts_source_tweets_and_reaction_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "mini.tar.bz2"
            src = {
                "id": 123,
                "text": "Official update from police",
                "created_at": "Fri Jan 09 12:56:25 +0000 2015",
                "retweet_count": 2,
                "favorite_count": 3,
                "user": {"followers_count": 100, "friends_count": 20, "verified": True},
            }
            reaction = {"id": 456, "text": "I deny this claim", "user": {}}
            with tarfile.open(archive, "w:bz2") as tf:
                for name, payload in [
                    ("pheme-rnr-dataset/demo/non-rumours/123/source-tweet/123.json", src),
                    ("pheme-rnr-dataset/demo/non-rumours/123/reactions/456.json", reaction),
                ]:
                    file_path = Path(tmp) / name.replace("/", "_")
                    file_path.write_text(json.dumps(payload), encoding="utf-8")
                    tf.add(file_path, arcname=name)
            docs = parse_pheme_archive(archive)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].event, "demo")
            self.assertEqual(docs[0].label, "non-rumour")
            self.assertEqual(docs[0].reaction_count, 1)
            self.assertEqual(docs[0].deny_count, 1)


class AttackTests(unittest.TestCase):
    def test_random_injection_uses_ratio_and_marks_attack(self):
        clean = [make_doc(f"c{i}") for i in range(4)]
        pool = [make_doc(f"r{i}", label="rumour") for i in range(5)]
        poisoned = build_random_poisoned_corpus(clean, pool, ratio=0.5, seed=7)
        injected = [d for d in poisoned if d.attack == "random"]
        self.assertEqual(len(poisoned), 6)
        self.assertEqual(len(injected), 2)
        self.assertTrue(all(d.label == "rumour" for d in injected))

    def test_semantic_injection_selects_most_similar_rumours(self):
        clean = [make_doc("c0")]
        pool = [
            make_doc("far", label="rumour", embedding=[0.0, 1.0]),
            make_doc("near", label="rumour", embedding=[1.0, 0.0]),
        ]
        poisoned = build_semantic_poisoned_corpus(clean, pool, query_embedding=[1.0, 0.0], ratio=1.0)
        injected_ids = [d.doc_id for d in poisoned if d.attack == "semantic"]
        self.assertEqual(injected_ids, ["near"])


class DefenseTests(unittest.TestCase):
    def test_semantic_outlier_filter_removes_far_cluster_member(self):
        docs = [
            make_doc("a", embedding=[1.0, 0.0]),
            make_doc("b", embedding=[0.98, 0.02]),
            make_doc("c", embedding=[0.99, 0.01]),
            make_doc("out", embedding=[-1.0, 0.0], label="rumour"),
        ]
        kept, audit = semantic_outlier_filter(docs, z_threshold=0.5, min_docs=3)
        self.assertNotIn("out", [d.doc_id for d in kept])
        self.assertEqual(audit["removed_count"], 1)

    def test_consistency_filter_removes_high_deny_documents(self):
        docs = [make_doc("ok", reaction_count=10, deny_count=0), make_doc("bad", reaction_count=10, deny_count=8)]
        kept, audit = consistency_filter(docs, deny_ratio_threshold=0.5)
        self.assertEqual([d.doc_id for d in kept], ["ok"])
        self.assertEqual(audit["removed_count"], 1)

    def test_social_rerank_prefers_verified_established_accounts(self):
        weak = make_doc("weak", followers_count=1, friends_count=1000, verified=False, created_at="Thu Jan 01 00:00:00 +0000 2015", default_profile=True)
        strong = make_doc("strong", followers_count=50000, friends_count=100, verified=True, listed_count=100, created_at="Thu Jan 01 00:00:00 +0000 2009")
        self.assertGreater(social_credibility_score(strong), social_credibility_score(weak))
        ranked = social_rerank([weak, strong], semantic_scores={"weak": 0.9, "strong": 0.8}, alpha=0.4)
        self.assertEqual(ranked[0].doc_id, "strong")


class RagAndMetricTests(unittest.TestCase):
    def test_vector_store_returns_cosine_sorted_results(self):
        store = InMemoryVectorStore([make_doc("x", embedding=[1.0, 0.0]), make_doc("y", embedding=[0.0, 1.0])])
        results = store.search([0.9, 0.1], top_k=2)
        self.assertEqual([r.doc.doc_id for r in results], ["x", "y"])
        self.assertGreater(results[0].score, results[1].score)

    def test_retrieval_metrics_reports_purity_and_poison_ranks(self):
        docs = [make_doc("a"), make_doc("b", label="rumour", attack="random"), make_doc("c")]
        metrics = retrieval_metrics(docs)
        self.assertEqual(metrics["retrieval_purity"], 2 / 3)
        self.assertEqual(metrics["poison_hit"], 1)
        self.assertEqual(metrics["avg_poison_rank"], 2.0)


if __name__ == "__main__":
    unittest.main()

class EcnuClientTests(unittest.TestCase):
    def test_embedding_payload_uses_ecnu_embedding_model_and_openai_shape(self):
        from rag_po.rag.ecnu_client import ECNUClient, ECNU_EMBEDDING_MODEL

        client = ECNUClient(api_key="test-key", base_url="https://example.test/v1")
        payload = client.embedding_payload(["hello", "world"])
        self.assertEqual(payload["model"], ECNU_EMBEDDING_MODEL)
        self.assertEqual(payload["input"], ["hello", "world"])

    def test_chat_payload_uses_supported_ecnu_chat_model_and_thinking_disabled(self):
        from rag_po.rag.ecnu_client import ECNUClient

        client = ECNUClient(api_key="test-key", chat_model="ecnu-plus", base_url="https://example.test/v1")
        payload = client.chat_payload([{"role": "user", "content": "hi"}], temperature=0.2)
        self.assertEqual(payload["model"], "ecnu-plus")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["temperature"], 0.2)

class DerivedMetricTests(unittest.TestCase):
    def test_generation_metrics_derives_asr_from_judge_scores(self):
        from rag_po.evaluation.metrics import generation_metrics_from_judge

        self.assertEqual(generation_metrics_from_judge({"misinfo_propagation": 2})["asr"], 1)
        self.assertEqual(generation_metrics_from_judge({"misinfo_propagation": 5})["asr"], 0)

    def test_defense_filter_metrics_counts_removed_poison_precision_and_recall(self):
        from rag_po.evaluation.metrics import defense_filter_metrics

        initial = [make_doc("clean"), make_doc("poison", label="rumour", attack="llm")]
        final = [make_doc("clean")]
        metrics = defense_filter_metrics(initial, final)
        self.assertEqual(metrics["filter_precision"], 1.0)
        self.assertEqual(metrics["filter_recall"], 1.0)

class VisualizationTests(unittest.TestCase):
    def test_build_html_report_renders_metrics_and_defense_chart(self):
        from pathlib import Path
        import tempfile
        from rag_po.io_utils import write_json
        from rag_po.evaluation.visualize import build_html_report

        rows = [
            {
                "event": "demo",
                "attack": "llm",
                "ratio": 0.1,
                "defense": "D0",
                "retrieval_metrics": {"retrieval_purity": 0.5, "poison_hit": 1},
                "generation_metrics": {"asr": 1},
                "defense_metrics": {"filter_precision": 0.0, "filter_recall": 0.0},
                "judge": {"overall_trustworthiness": 2, "factual_accuracy": 3, "misinfo_propagation": 2, "uncertainty_expression": 1},
            },
            {
                "event": "demo",
                "attack": "llm",
                "ratio": 0.1,
                "defense": "D_all",
                "retrieval_metrics": {"retrieval_purity": 1.0, "poison_hit": 0},
                "generation_metrics": {"asr": 0},
                "defense_metrics": {"filter_precision": 1.0, "filter_recall": 1.0},
                "judge": {"overall_trustworthiness": 5, "factual_accuracy": 5, "misinfo_propagation": 5, "uncertainty_expression": 5},
            },
        ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            results = Path(tmp) / "results.json"
            write_json(results, rows)
            report = build_html_report(results)
            html = report.read_text(encoding="utf-8")
        self.assertIn("RAG Poisoning Results", html)
        self.assertIn("D_all", html)
        self.assertIn("Retrieval Purity", html)
        self.assertIn("<svg", html)
        self.assertIn("filter_recall", html)

class VisualizationCliTests(unittest.TestCase):
    def test_cli_visualize_writes_html_report(self):
        from pathlib import Path
        import tempfile
        from rag_po.cli import main
        from rag_po.io_utils import write_json

        rows = [{
            "event": "demo",
            "attack": "llm",
            "ratio": 0.1,
            "defense": "D_all",
            "retrieval_metrics": {"retrieval_purity": 1.0, "poison_hit": 0},
            "generation_metrics": {"asr": 0},
            "defense_metrics": {"filter_precision": 1.0, "filter_recall": 1.0},
            "judge": {"overall_trustworthiness": 5, "factual_accuracy": 5, "misinfo_propagation": 5, "uncertainty_expression": 5},
        }]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            results = Path(tmp) / "results.json"
            out = Path(tmp) / "viz.html"
            write_json(results, rows)
            code = main(["visualize", str(results), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())
            self.assertIn("RAG Poisoning Results", out.read_text(encoding="utf-8"))

class ArchiveFormatTests(unittest.TestCase):
    def test_parse_archive_accepts_gzip_tar_even_with_bz2_suffix(self):
        import json
        import tarfile
        import tempfile
        from pathlib import Path
        from rag_po.data_prep.parse_pheme import parse_pheme_archive

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            archive = Path(tmp) / "mini.tar.bz2"
            payload = {"id": 1, "text": "A gzip tar source tweet", "user": {}}
            item = Path(tmp) / "one.json"
            item.write_text(json.dumps(payload), encoding="utf-8")
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(item, arcname="pheme-rnr-dataset/demo/rumours/1/source-tweet/1.json")
            docs = parse_pheme_archive(archive)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].label, "rumour")

class Pheme9LayoutTests(unittest.TestCase):
    def test_parse_archive_accepts_all_rnr_annotated_threads_layout(self):
        import json
        import tarfile
        import tempfile
        from pathlib import Path
        from rag_po.data_prep.parse_pheme import parse_pheme_archive

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            archive = Path(tmp) / "mini9.tar.gz"
            src = {"id": 10, "text": "PHEME nine layout source", "user": {}}
            reaction = {"id": 11, "text": "this is false", "user": {}}
            for filename, payload in [("src.json", src), ("reaction.json", reaction)]:
                (Path(tmp) / filename).write_text(json.dumps(payload), encoding="utf-8")
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(Path(tmp) / "src.json", arcname="all-rnr-annotated-threads/gurlitt-all-rnr-threads/rumours/10/source-tweets/10.json")
                tf.add(Path(tmp) / "reaction.json", arcname="all-rnr-annotated-threads/gurlitt-all-rnr-threads/rumours/10/reactions/11.json")
            docs = parse_pheme_archive(archive)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].event, "gurlitt")
        self.assertEqual(docs[0].reaction_count, 1)
        self.assertEqual(docs[0].deny_count, 1)

class CliPrepareArgsTests(unittest.TestCase):
    def test_negative_max_per_event_label_means_unlimited_for_prepare(self):
        from argparse import Namespace
        from rag_po.cli import prepare_limit_from_args

        self.assertIsNone(prepare_limit_from_args(Namespace(max_per_event_label=-1)))
        self.assertEqual(prepare_limit_from_args(Namespace(max_per_event_label=3)), 3)

class LeakageAndAblationTests(unittest.TestCase):
    def test_generation_context_does_not_include_ground_truth_label_or_attack(self):
        from rag_po.rag.generator import build_context

        doc = make_doc("p1", label="rumour", attack="llm", text="A suspicious claim")
        context = build_context([doc])
        self.assertNotIn("label=", context)
        self.assertNotIn("RUMOUR", context)
        self.assertNotIn("NON_RUMOUR", context)
        self.assertNotIn("attack", context.lower())
        self.assertIn("text=A suspicious claim", context)

    def test_consistency_score_does_not_use_attack_ground_truth(self):
        from rag_po.defense.consistency import document_suspicion_score

        plain = make_doc("plain", text="Unconfirmed source claims the crash timeline changed")
        attacked = make_doc("attacked", text="Unconfirmed source claims the crash timeline changed", attack="llm")
        self.assertEqual(document_suspicion_score(plain), document_suspicion_score(attacked))

    def test_consistency_filter_removes_numeric_and_outcome_conflict_against_majority(self):
        docs = [
            make_doc("m1", text="Officials say Flight 4U9525 crashed in the French Alps with 150 people onboard."),
            make_doc("m2", text="Police confirm Flight 4U9525 crashed in the French Alps with 150 people onboard."),
            make_doc("bad", text="Witnesses say Flight 4U9525 safely landed in Nice with 0 casualties."),
        ]
        kept, audit = consistency_filter(docs)
        self.assertEqual([doc.doc_id for doc in kept], ["m1", "m2"])
        self.assertIn("bad", audit["removed_ids"])
        self.assertGreater(audit["suspicion_scores"]["bad"], audit["suspicion_scores"]["m1"])

    def test_defense_specs_include_requested_ablation_combinations(self):
        from rag_po.models import DEFENSE_SPECS

        self.assertTrue(DEFENSE_SPECS["D123"].outlier)
        self.assertTrue(DEFENSE_SPECS["D123"].consistency)
        self.assertTrue(DEFENSE_SPECS["D123"].social)
        self.assertFalse(DEFENSE_SPECS["D123"].critical_prompt)
        self.assertEqual((DEFENSE_SPECS["D34"].social, DEFENSE_SPECS["D34"].critical_prompt), (True, True))
        self.assertFalse(DEFENSE_SPECS["D34"].outlier)
        self.assertEqual(
            (DEFENSE_SPECS["D234"].consistency, DEFENSE_SPECS["D234"].social, DEFENSE_SPECS["D234"].critical_prompt),
            (True, True, True),
        )
