"""Web UI for Spotify Web authorization.

This module exposes the same PKCE flow that ``mopidy spotify auth web`` drives
from a terminal as a small web app, mounted under ``/spotify/auth`` on Mopidy's
HTTP server. The browser receives the authorization URL, the user completes
authorization at Spotify, and pastes the callback result displayed by the
redirect page back into the page. The server then validates the state,
exchanges the code, and persists the refresh token exactly as the CLI does.

Pending challenges live in memory only, mirroring the CLI contract that the
PKCE verifier and CSRF state are never persisted. The refresh-token storage
policy comes from ``[spotify] auth_storage`` so the choice stays with the
operator, not the browser user.

Mopidy's HTTP server is unauthenticated LAN-trust, like the core HTTP API.
The POST endpoints additionally require a JSON body and a same-origin browser
context to limit cross-site request forgery.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import tornado.web
from tornado.ioloop import IOLoop

from mopidy_spotify import Extension
from mopidy_spotify.oauth import flow, manifest, store

if TYPE_CHECKING:
    from collections.abc import Callable

    from mopidy.config import Config
    from mopidy.core import CoreProxy

logger = logging.getLogger(__name__)

FLOW_TTL_SECONDS = 300.0


class FlowNotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("No pending authorization flow, please start again.")


@dataclass(frozen=True)
class _PendingChallenge:
    challenge: flow.AuthChallenge
    expires_at: float


class PendingFlows:
    """Single in-memory slot holding at most one unfinished challenge.

    A new flow replaces any pending challenge, and finishing pops the entry so
    a flow id can only be redeemed once. The verifier and CSRF state are never
    persisted.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = FLOW_TTL_SECONDS,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._time = time_func
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingChallenge] = {}

    def create(self, challenge: flow.AuthChallenge) -> tuple[str, float]:
        flow_id = secrets.token_urlsafe(32)
        with self._lock:
            self._pending.clear()
            self._pending[flow_id] = _PendingChallenge(
                challenge=challenge,
                expires_at=self._time() + self._ttl,
            )
        return flow_id, self._ttl

    def pop(self, flow_id: str) -> flow.AuthChallenge | None:
        with self._lock:
            pending = self._pending.pop(flow_id, None)
        if pending is None:
            return None
        if self._time() >= pending.expires_at:
            return None
        return pending.challenge


class AuthService:
    """Coordinates the web-driven authorization flow for one Mopidy instance."""

    def __init__(self, config: Config, core: CoreProxy | None = None) -> None:
        self._config = config
        self._core = core
        self._store = store.Store(Extension.get_auth_state_path(config))
        self._pending = PendingFlows()

    def status(self) -> dict[str, Any]:
        """Report the persisted authorization state without resolving secrets."""
        try:
            stored = self._store.load_manifest()
        except (store.InvalidManifestError, store.Error):
            return {"mode": None, "state": "invalid", "storage": None}
        if stored is None:
            return {"mode": None, "state": "unauthorized", "storage": None}
        storage = (
            stored.refresh_token.storage
            if isinstance(stored, manifest.PkceAuthorized)
            else None
        )
        return {"mode": stored.mode, "state": stored.state, "storage": storage}

    def start(self) -> dict[str, Any]:
        challenge = self._build_flow().start_auth()
        flow_id, expires_in = self._pending.create(challenge)
        logger.info("Started Spotify Web authorization from the web UI")
        return {
            "flow_id": flow_id,
            "authorization_url": challenge.authorization_url,
            "expires_in": int(expires_in),
        }

    def finish(self, flow_id: str, pasted_result: str) -> None:
        """Complete a pending flow; blocking, so callers must offload it."""
        challenge = self._pending.pop(flow_id)
        if challenge is None:
            raise FlowNotFoundError
        self._build_flow().finish_auth(challenge, pasted_result)
        logger.info("Stored Spotify Web authorization from the web UI")
        self._notify_backend()

    def _build_flow(self) -> flow.AuthFlow:
        policy = flow.StoragePolicy(self._config["spotify"]["auth_storage"])
        writer = flow.authorization_writer(self._store, policy)
        return flow.AuthFlow(self._config, persist_authorization=writer)

    def _notify_backend(self) -> None:
        """Ask the Spotify backend to re-login so playlists appear without restart."""
        if self._core is None:
            return
        try:
            # ``backends`` and ``refresh_login`` are resolved dynamically through
            # pykka proxies and are not declared on the typed proxy classes.
            backends = getattr(self._core, "backends").get()  # noqa: B009
            for backend in backends:
                if "spotify" in backend.uri_schemes.get():
                    getattr(backend, "refresh_login").defer()  # noqa: B009
                    return
        except Exception:
            logger.warning(
                "Spotify was authorized via the web UI, but the backend could not "
                "be notified; restart Mopidy to load playlists.",
                exc_info=True,
            )


class _AuthHandler(tornado.web.RequestHandler):
    def initialize(self, service: AuthService) -> None:
        self.service = service

    def set_default_headers(self) -> None:
        self.set_header("Cache-Control", "no-store")

    def _reject_cross_origin_post(self) -> bool:
        # A JSON content type forces a CORS preflight that we never answer, and
        # the origin checks reject simple cross-site form posts. Non-browser
        # clients may omit both headers.
        content_type = self.request.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            return True
        origin = self.request.headers.get("Origin")
        if origin is not None:
            return urlsplit(origin).netloc != self.request.host
        sec_fetch_site = self.request.headers.get("Sec-Fetch-Site")
        if sec_fetch_site is not None:
            return sec_fetch_site not in ("same-origin", "none")
        return False

    def _write_forbidden(self) -> None:
        self.set_status(403)
        self.write({"error": "Cross-origin requests are not allowed."})


class PageHandler(_AuthHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "text/html; charset=UTF-8")
        self.write(_PAGE_HTML)


class StatusHandler(_AuthHandler):
    def get(self) -> None:
        self.write(self.service.status())


class StartHandler(_AuthHandler):
    def post(self) -> None:
        if self._reject_cross_origin_post():
            self._write_forbidden()
            return
        self.write(self.service.start())


class FinishHandler(_AuthHandler):
    async def post(self) -> None:
        if self._reject_cross_origin_post():
            self._write_forbidden()
            return

        try:
            payload = json.loads(self.request.body)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            self.set_status(400)
            self.write({"error": "Invalid JSON body."})
            return

        flow_id = payload.get("flow_id")
        result = payload.get("result")
        if not isinstance(flow_id, str) or not isinstance(result, str):
            self.set_status(400)
            self.write({"error": "flow_id and result are required."})
            return

        try:
            # Token exchange and persistence block, so keep them off the
            # Tornado event loop.
            await IOLoop.current().run_in_executor(
                None,
                self.service.finish,
                flow_id,
                result,
            )
        except (FlowNotFoundError, flow.AuthFlowError) as exc:
            self.set_status(400)
            self.write({"error": str(exc)})
            return

        self.write({"status": "authorized"})


def make_auth_app(
    config: Config,
    core: CoreProxy,
) -> list[tuple[str, type[tornado.web.RequestHandler], dict[str, Any]]]:
    """Build the ``http:app`` request rules for Spotify Web authorization."""
    service = AuthService(config, core)
    return [
        (r"/auth/?", PageHandler, {"service": service}),
        (r"/auth/status", StatusHandler, {"service": service}),
        (r"/auth/start", StartHandler, {"service": service}),
        (r"/auth/finish", FinishHandler, {"service": service}),
    ]


_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spotify login</title>
</head>
<body>
<h1>Spotify</h1>
<p id="status">Checking status&hellip;</p>
<div id="login" hidden>
<button id="start">Log in with Spotify</button>
</div>
<div id="flow" hidden>
<p><a id="authlink" target="_blank" rel="noopener">Open the Spotify
authorization page</a></p>
<p>After authorizing, copy the result shown in your browser and paste it
here:</p>
<p><textarea id="result" rows="4" cols="60"></textarea></p>
<button id="finish">Complete login</button>
</div>
<p id="message"></p>
<script>
const $ = (id) => document.getElementById(id);
async function refreshStatus() {
  const response = await fetch("status");
  const status = await response.json();
  if (status.mode === "pkce" && status.state === "authorized") {
    $("status").textContent = "Authorized" +
      (status.storage ? " (token stored as " + status.storage + ")" : "");
    $("login").hidden = true;
    $("flow").hidden = true;
  } else {
    $("status").textContent = status.state === "invalid"
      ? "Stored authorization state is invalid"
      : "Not authorized";
    $("login").hidden = false;
  }
}
$("start").addEventListener("click", async () => {
  const response = await fetch("start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const start = await response.json();
  if (!response.ok) {
    $("message").textContent = start.error || "Could not start login.";
    return;
  }
  window.flowId = start.flow_id;
  $("authlink").href = start.authorization_url;
  $("message").textContent = "";
  $("login").hidden = true;
  $("flow").hidden = false;
});
$("finish").addEventListener("click", async () => {
  const response = await fetch("finish", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ flow_id: window.flowId, result: $("result").value }),
  });
  const result = await response.json();
  if (response.ok) {
    $("message").textContent = "Login complete.";
    await refreshStatus();
  } else {
    $("message").textContent = result.error || "Login failed.";
  }
});
refreshStatus();
</script>
</body>
</html>
"""
