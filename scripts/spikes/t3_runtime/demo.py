"""Run the adapter against a scripted loopback HTTP service. No provider runs."""

from collections.abc import Generator
from contextlib import closing, contextmanager
from http.client import HTTPConnection, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import cast, final

import msgspec
from typing_extensions import override

from .backend import Journal, T3Backend


class ToolCall(msgspec.Struct, frozen=True):
    name: str
    arguments: dict[str, str | int]


@final
class Fixture:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.runs: dict[str, int] = {}
        self.status = "completed"
        self.drop_reply = True

    def invoke(self, call: ToolCall) -> object:
        self.calls.append(call)
        args = call.arguments
        if call.name == "t3_thread_launch":
            serial = self.runs.setdefault(str(args["clientRequestId"]), len(self.runs) + 1)
            if self.drop_reply:
                self.drop_reply = False
                return None
            return {
                "projectId": args["projectId"],
                "threadId": f"thread-{serial}",
                "runId": f"run-{serial}",
                "status": self.status,
            }
        if call.name == "orchestrator_capabilities":
            return {"features": {"cancellation": True}}
        thread = str(args["threadId"])
        run = thread.replace("thread-", "run-")
        if call.name == "t3_thread_read":
            return {
                "thread": {"threadId": thread},
                "recentRuns": [{"runId": run, "status": self.status}],
            }
        if args["runId"] != run:
            raise ValueError("Wrong run")
        if call.name == "t3_thread_wait":
            return {
                "threadId": thread,
                "runId": run,
                "status": self.status,
                "timedOut": False,
            }
        if call.name == "t3_thread_interrupt" and "stop" not in args:
            return {"threadId": thread, "runId": run, "status": "interrupt_requested"}
        raise ValueError(call.name)


@contextmanager
def serve(fixture: Fixture) -> Generator[T3Backend, None, None]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            result = fixture.invoke(msgspec.json.decode(body, type=ToolCall))
            if result is None:
                return
            self.send_response(200)
            self.end_headers()
            _ = self.wfile.write(msgspec.json.encode(result))

        @override
        def log_message(self, format: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()

        def call(name: str, args: dict[str, str | int]) -> object:
            with closing(HTTPConnection("127.0.0.1", server.server_port, timeout=10)) as client:
                client.request("POST", "/", msgspec.json.encode(ToolCall(name, args)))
                with client.getresponse() as reply:
                    if reply.status != 200:
                        raise RuntimeError(f"Fixture HTTP status {reply.status}")
                    return cast(object, msgspec.json.decode(reply.read()))

        try:
            yield T3Backend("fixture-environment", "fixture-project", call)
        finally:
            server.shutdown()
            worker.join()


def main() -> None:
    with TemporaryDirectory() as directory, serve(Fixture()) as backend:
        path = Path(directory) / "attempts.db"
        journal = Journal(path, backend)
        journal.initialize()
        try:
            _ = journal.start("A")
        except RemoteDisconnected:
            print("Fixture: launch reply lost")
        journal = Journal(path, backend)
        a = journal.start("A")
        print(f"Fixture: recovered A ({backend.wait(a.ref).status})")
        journal.accept(a)
        print("Fixture: A accepted")
        b = journal.start("B")
        print(f"Fixture: B started ({b.ref.run})")
        journal.accept(b)
        print("Fixture: B accepted")


if __name__ == "__main__":
    main()
