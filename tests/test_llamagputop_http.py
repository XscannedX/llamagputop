import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import llamagputop_http as api


class ApiAdapterTests(unittest.TestCase):
    def test_model_stats_maps_live_and_speculative_values(self):
        result = api.model_stats({
            "pp": 24.5,
            "tg": 28.7,
            "tg_life": 26.1,
            "reasoning_format": None,
            "spec": 0.72,
            "tok_step": 1.72,
            "spec_acc": 86,
        })
        self.assertEqual(result, {
            "prefill": 24.5,
            "gen": 28.7,
            "session-avg": 26.1,
            "reasoning": "none",
            "draft-accepted-p": 0.72,
            "draft-accepted-tok": 1.72,
            "draft-accepted-total": 86,
        })

    def test_model_config_preserves_unavailable_values_as_null(self):
        result = api.model_config(
            {"ctx_total": 131072, "slots": 1, "spec_type": None},
            {"loading": [("ngl", "33")], "sampling": [("temp", "0.6")]},
        )
        self.assertEqual(result["ctx"], 131072)
        self.assertEqual(result["ngl"], 33)
        self.assertEqual(result["temp"], 0.6)
        self.assertEqual(result["slots"], 1)
        self.assertIsNone(result["top-k"])
        self.assertEqual(result["spec-type"], "none")

    def test_model_config_uses_command_line_slots_when_runtime_is_missing(self):
        result = api.model_config(
            {"ctx": 4096, "slots": None, "spec_type": "none"},
            {"loading": [("slots", "2")]},
        )
        self.assertEqual(result["slots"], 2)

    def test_model_config_falls_back_to_raw_kv_flags(self):
        original = api._command_for_port
        api._command_for_port = lambda port: [
            "llama-server", "--port", str(port),
            "-ctk", "q5_1", "-ctv", "q5_1",
            "-ctkd", "q8_0", "-ctvd", "q8_0",
        ]
        try:
            result = api.model_config(
                {"ctx": 4096, "slots": 1, "spec_type": "none"},
                {}, port="7679",
            )
        finally:
            api._command_for_port = original
        self.assertEqual(result["kv-k/v"], "q5_1/q5_1")
        self.assertEqual(result["draft-kv"], "q8_0/q8_0")

    def test_collect_snapshot_uses_monitor_collection_without_feed(self):
        original = api.monitor._collect
        api.monitor._collect = lambda gpus, port: (
            [], {}, {}, [{"port": port, "alive": False}], {}, [],
        )
        try:
            snapshot = api.collect_snapshot([], "7679")
        finally:
            api.monitor._collect = original
        self.assertEqual(snapshot["llama"]["port"], "7679")
        self.assertFalse(snapshot["llama"]["alive"])

    def test_snapshot_selects_the_configured_server(self):
        snapshot = api.build_snapshot(
            [{"vendor": "AMD", "util": 12}],
            {"util": 4},
            {"free": 100, "total": 200},
            [{"port": "8080", "alive": True, "model": "model.gguf", "tg": 3}],
            {"8080": {"loading": [("ctx", "131072")]}},
            [],
        )
        self.assertEqual(snapshot["llama"]["port"], "8080")
        self.assertEqual(snapshot["llama"]["model"], "model.gguf")
        self.assertEqual(snapshot["modelStats"]["gen"], 3)
        self.assertEqual(snapshot["modelConfig"]["ctx"], 131072)


class HttpEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = api.SnapshotStore(port="8080")
        cls.store.publish({
            "updatedAt": "now",
            "llama": {"port": "8080", "alive": True, "stale": False},
            "modelStats": {"gen": 3},
            "modelConfig": {"ctx": 131072},
        })
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), api.handler_for(cls.store))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def get(self, path):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        body = json.loads(response.read())
        conn.close()
        return response.status, body

    def test_health_reports_live_wrapper_and_target(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["llama"]["alive"])

    def test_stats_and_config_return_cached_snapshot_sections(self):
        status, stats = self.get("/stats")
        self.assertEqual(status, 200)
        self.assertEqual(stats["modelStats"]["gen"], 3)

        status, config = self.get("/config")
        self.assertEqual(status, 200)
        self.assertEqual(config["modelConfig"]["ctx"], 131072)

    def test_unknown_path_is_not_an_endpoint(self):
        status, _ = self.get("/missing")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
