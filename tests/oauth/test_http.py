import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import tornado.web
from pydantic import SecretStr
from tornado.testing import AsyncHTTPTestCase

from mopidy_spotify import Extension
from mopidy_spotify.oauth import flow, http, manifest, store
from mopidy_spotify.oauth.http import PendingFlows, make_auth_app


def make_config(data_dir: Path, auth_storage: str = "auto") -> dict[str, Any]:
    return {
        "core": {"data_dir": str(data_dir)},
        "proxy": {},
        "spotify": {"timeout": 10, "auth_storage": auth_storage},
    }


def stub_challenge() -> flow.AuthChallenge:
    return flow.AuthChallenge(
        authorization_url="https://accounts.spotify.com/authorize?x=1",
        state="state-123",
        verifier="verifier-123",
    )


class WebAuthTestCase(AsyncHTTPTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.config = make_config(self.data_dir)
        self.auth_state_path = Extension.get_auth_state_path(self.config)
        self.core = mock.Mock()
        super().setUp()

    def get_app(self):
        return tornado.web.Application(make_auth_app(self.config, self.core))

    def same_origin(self) -> str:
        return f"http://127.0.0.1:{self.get_http_port()}"

    def post_json(self, path: str, payload: Any = None, **headers: str):
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("Origin", self.same_origin())
        return self.fetch(
            path,
            method="POST",
            body=json.dumps(payload if payload is not None else {}).encode(),
            headers=headers,
        )


class StatusTests(WebAuthTestCase):
    def test_status_without_manifest_is_unauthorized(self):
        response = self.fetch("/auth/status")
        assert response.code == 200
        assert json.loads(response.body) == {
            "mode": None,
            "state": "unauthorized",
            "storage": None,
        }

    def test_status_reports_inline_storage(self):
        store.Store(self.auth_state_path).persist_pkce_authorization(
            SecretStr("refresh-token"),
            manifest.StorageType.INLINE,
        )
        status = json.loads(self.fetch("/auth/status").body)
        assert status == {"mode": "pkce", "state": "authorized", "storage": "inline"}

    def test_status_reports_keyring_descriptor_without_keyring_access(self):
        self.auth_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "mode": "pkce",
                    "state": "authorized",
                    "refresh_token": {
                        "storage": "keyring",
                        "service": "mopidy-spotify",
                        "username": "user-123",
                    },
                }
            ),
            encoding="utf-8",
        )
        status = json.loads(self.fetch("/auth/status").body)
        assert status == {"mode": "pkce", "state": "authorized", "storage": "keyring"}

    def test_status_reports_bridge(self):
        store.Store(self.auth_state_path).configure_bridge()
        status = json.loads(self.fetch("/auth/status").body)
        assert status == {"mode": "bridge", "state": "configured", "storage": None}

    def test_status_reports_cleared(self):
        store.Store(self.auth_state_path).clear()
        status = json.loads(self.fetch("/auth/status").body)
        assert status == {"mode": "bridge", "state": "cleared", "storage": None}

    def test_status_reports_invalid_manifest(self):
        self.auth_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_state_path.write_text("not json", encoding="utf-8")
        status = json.loads(self.fetch("/auth/status").body)
        assert status == {"mode": None, "state": "invalid", "storage": None}

    def test_status_never_leaks_refresh_token(self):
        store.Store(self.auth_state_path).persist_pkce_authorization(
            SecretStr("super-secret-token"),
            manifest.StorageType.INLINE,
        )
        assert "super-secret-token" not in self.fetch("/auth/status").body.decode()


class PageTests(WebAuthTestCase):
    def test_page_is_served(self):
        response = self.fetch("/auth/")
        assert response.code == 200
        assert response.headers["Content-Type"].startswith("text/html")
        assert b"Log in with Spotify" in response.body

    def test_page_served_without_trailing_slash(self):
        response = self.fetch("/auth")
        assert response.code == 200


class RootRedirectTests(WebAuthTestCase):
    def test_root_redirects_to_auth_page(self):
        response = self.fetch("/", follow_redirects=False)
        assert response.code == 302
        assert response.headers["Location"] == "/auth/"

    def test_root_redirect_target_resolves_to_page(self):
        response = self.fetch("/")
        assert response.code == 200
        assert b"Log in with Spotify" in response.body


class StartFinishTests(WebAuthTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(http.flow, "AuthFlow")
        self.flow_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.flow = self.flow_cls.return_value
        self.flow.start_auth.return_value = stub_challenge()

    def configure_backend_notification(self):
        backend = mock.Mock()
        backend.uri_schemes.get.return_value = ["spotify"]
        self.core.backends.get.return_value = [backend]
        return backend

    def start_flow(self) -> str:
        response = self.post_json("/auth/start")
        assert response.code == 200
        return json.loads(response.body)["flow_id"]

    def test_start_returns_challenge(self):
        response = self.post_json("/auth/start")
        assert response.code == 200
        payload = json.loads(response.body)
        assert payload["authorization_url"] == (
            "https://accounts.spotify.com/authorize?x=1"
        )
        assert payload["expires_in"] == 300
        assert len(payload["flow_id"]) > 20

    def test_start_replaces_previous_flow(self):
        first = self.start_flow()
        second = self.start_flow()
        response = self.post_json(
            "/auth/finish",
            {"flow_id": first, "result": "code=abc&state=state-123"},
        )
        assert response.code == 400
        assert "No pending authorization flow" in json.loads(response.body)["error"]
        assert second != first

    def test_finish_success_notifies_backend(self):
        flow_id = self.start_flow()
        backend = self.configure_backend_notification()

        response = self.post_json(
            "/auth/finish",
            {"flow_id": flow_id, "result": "code=abc&state=state-123"},
        )

        assert response.code == 200
        assert json.loads(response.body) == {"status": "authorized"}
        self.flow.finish_auth.assert_called_once_with(
            stub_challenge(),
            "code=abc&state=state-123",
        )
        backend.refresh_login.defer.assert_called_once()

    def test_finish_succeeds_when_backend_notification_fails(self):
        flow_id = self.start_flow()
        self.core.backends.get.side_effect = RuntimeError("no backends")

        response = self.post_json(
            "/auth/finish",
            {"flow_id": flow_id, "result": "code=abc&state=state-123"},
        )

        assert response.code == 200

    def test_finish_without_pending_flow(self):
        response = self.post_json(
            "/auth/finish",
            {"flow_id": "unknown", "result": "code=abc&state=state-123"},
        )
        assert response.code == 400
        assert "No pending authorization flow" in json.loads(response.body)["error"]

    def test_finish_maps_flow_errors_to_bad_request(self):
        flow_id = self.start_flow()
        self.flow.finish_auth.side_effect = flow.AuthStateMismatchError()

        response = self.post_json(
            "/auth/finish",
            {"flow_id": flow_id, "result": "code=abc&state=wrong"},
        )

        assert response.code == 400
        assert "Incorrect state" in json.loads(response.body)["error"]

    def test_finish_rejects_non_json_body(self):
        response = self.fetch(
            "/auth/finish",
            method="POST",
            body=b"not json",
            headers={
                "Content-Type": "application/json",
                "Origin": self.same_origin(),
            },
        )
        assert response.code == 400

    def test_finish_rejects_missing_fields(self):
        response = self.post_json("/auth/finish", {"flow_id": "abc"})
        assert response.code == 400


class CsrfTests(WebAuthTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(http.flow, "AuthFlow")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.flow_cls = http.flow.AuthFlow
        self.flow_cls.return_value.start_auth.return_value = stub_challenge()

    def test_cross_origin_post_is_rejected(self):
        response = self.post_json(
            "/auth/start",
            Origin="http://evil.example",
        )
        assert response.code == 403

    def test_non_json_content_type_is_rejected(self):
        response = self.fetch(
            "/auth/start",
            method="POST",
            body=b"{}",
            headers={"Content-Type": "text/plain", "Origin": self.same_origin()},
        )
        assert response.code == 403

    def test_cross_site_fetch_is_rejected(self):
        response = self.fetch(
            "/auth/start",
            method="POST",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Sec-Fetch-Site": "cross-site",
            },
        )
        assert response.code == 403

    def test_same_origin_post_is_allowed(self):
        response = self.fetch(
            "/auth/start",
            method="POST",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Sec-Fetch-Site": "same-origin",
            },
        )
        assert response.code == 200

    def test_non_browser_post_is_allowed(self):
        response = self.fetch(
            "/auth/start",
            method="POST",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert response.code == 200


class PendingFlowsTests:
    def test_pop_returns_challenge_once(self):
        flows = PendingFlows()
        challenge = stub_challenge()
        flow_id, ttl = flows.create(challenge)

        assert ttl == http.FLOW_TTL_SECONDS
        assert flows.pop(flow_id) is challenge
        assert flows.pop(flow_id) is None

    def test_expired_flow_is_dropped(self):
        now = [1000.0]
        flows = PendingFlows(time_func=lambda: now[0])
        flow_id, _ = flows.create(stub_challenge())

        now[0] += http.FLOW_TTL_SECONDS + 1

        assert flows.pop(flow_id) is None

    def test_new_flow_replaces_pending_flow(self):
        flows = PendingFlows()
        first, _ = flows.create(stub_challenge())
        second, _ = flows.create(stub_challenge())

        assert flows.pop(first) is None
        assert flows.pop(second) is not None


class StoragePolicyTests:
    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            ("auto", flow.StoragePolicy.AUTO),
            ("plaintext", flow.StoragePolicy.PLAINTEXT),
            ("keyring", flow.StoragePolicy.KEYRING),
        ],
    )
    def test_build_flow_uses_configured_policy(
        self,
        tmp_path: Path,
        configured: str,
        expected: flow.StoragePolicy,
    ):
        service = http.AuthService(make_config(tmp_path, auth_storage=configured))

        with (
            mock.patch.object(http.flow, "authorization_writer") as writer,
            mock.patch.object(http.flow, "AuthFlow"),
        ):
            service.start()

        assert writer.call_args.args[1] is expected
