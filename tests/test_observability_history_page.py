import importlib
from html.parser import HTMLParser

from fastapi.testclient import TestClient

from pickparts_agent.observability import NullRecorder
from test_observability_runs_on_worker import ScriptedBackend


def test_history_page_assets_and_index_link():
    web = importlib.import_module("pickparts_agent.interfaces.web")
    app = web.create_app(factory=ScriptedBackend, recorder=NullRecorder())
    with TestClient(app) as client:
        page = client.get("/history")
        assert page.status_code == 200
        assert "run-list" in page.text and "span-pane" in page.text

        js = client.get("/static/history.js")
        assert js.status_code == 200
        assert "/api/runs" in js.text and "artifacts" in js.text

        index = client.get("/")
        assert index.status_code == 200 and 'href="/history"' in index.text


def test_served_pages_expose_optimized_filter_and_separate_ordered_subtasks():
    class Controls(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = {}
            self.options = {}
            self.select = self.option = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.elements[attrs["id"]] = (tag, attrs)
            if tag == "select":
                self.select = attrs.get("id")
            elif tag == "option":
                self.option = (self.select, attrs.get("value"))
                self.options[self.option] = ""

        def handle_data(self, data):
            if self.option:
                self.options[self.option] += data

        def handle_endtag(self, tag):
            if tag == "option":
                self.option = None
            elif tag == "select":
                self.select = None

    web = importlib.import_module("pickparts_agent.interfaces.web")
    with TestClient(web.create_app(factory=ScriptedBackend, recorder=NullRecorder())) as client:
        history, console = Controls(), Controls()
        history.feed(client.get("/history").text)
        console.feed(client.get("/").text)
        assert history.options[("run-agent", "tiptop_optimized")] == "TiPToP 优化版"
        assert history.options[("run-agent", "tiptop")] == "TiPToP 初版"
        assert console.elements["subtask-lane"][0] == "ol"
        assert console.elements["subtask-progress"][1]["role"] == "status"
        assert console.elements["workflow-lane"][1]["aria-label"] != console.elements["subtask-lane"][1]["aria-label"]
