# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.models.model_error import ModelError, ModelErrorKind
from openviking.server.model_connection_test import probe_model


@pytest.fixture
def vlm(monkeypatch):
    instance = Mock(get_completion_async=AsyncMock(return_value="OK"))
    factory = Mock(return_value=instance)
    monkeypatch.setattr(
        "openviking_cli.utils.config.vlm_config.VLMConfig.get_vlm_instance", factory
    )
    return instance


async def test_vlm_probe_uses_one_draft_credential_and_closes(vlm, monkeypatch):
    captured = []
    from openviking_cli.utils.config.vlm_config import VLMConfig

    def build(settings):
        captured.append(settings)
        return vlm

    monkeypatch.setattr(VLMConfig, "get_vlm_instance", build)
    fallback = {
        "model": "o3",
        "max_tokens": 4096,
        "credentials": [{"provider": "openai", "api_key": "other"}],
    }
    result = await probe_model("vlm", {"provider": "openai", "api_key": "draft"}, fallback, True)
    assert result["ok"]
    assert len(captured[0].credentials) == 1
    assert captured[0].credentials[0].api_key == "draft"
    assert captured[0].thinking is True
    assert captured[0].max_tokens == 4096
    assert fallback["credentials"][0]["api_key"] == "other"
    vlm.close.assert_called_once()


async def test_probe_redacts_provider_errors(vlm):
    vlm.get_completion_async.side_effect = ModelError(
        ModelErrorKind.AUTHENTICATION_ERROR, "secret-token"
    )
    result = await probe_model(
        "vlm", {"provider": "openai", "api_key": "secret-token"}, {"model": "o3"}
    )
    assert result == {"ok": False, "code": "authentication_error"}
    assert "secret-token" not in str(result)
    vlm.close.assert_called_once()


async def test_probe_timeout_closes_model(vlm):
    vlm.get_completion_async.side_effect = TimeoutError()
    assert await probe_model(
        "vlm", {"provider": "openai", "api_key": "draft"}, {"model": "o3"}
    ) == {"ok": False, "code": "timeout"}
    vlm.close.assert_called_once()


async def test_probe_deadline_cancels_request_and_closes(vlm, monkeypatch):
    wait_for = asyncio.wait_for
    monkeypatch.delattr(asyncio, "timeout", raising=False)

    async def short_deadline(request, timeout):
        assert timeout == 25
        return await wait_for(request, timeout=0.001)

    async def pending(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "wait_for", short_deadline)
    vlm.get_completion_async.side_effect = pending
    assert await probe_model(
        "vlm", {"provider": "openai", "api_key": "draft"}, {"model": "o3"}
    ) == {"ok": False, "code": "timeout"}
    vlm.close.assert_called_once()


@pytest.mark.parametrize(
    "provider", ["jina", "gemini", "voyage", "dashscope", "minimax", "cohere", "litellm"]
)
async def test_embedding_api_key_providers_use_existing_adapter(provider, monkeypatch):
    instance = Mock(
        embed_async=AsyncMock(return_value=Mock(dense_vector=[0.1], sparse_vector=None))
    )

    def build(settings):
        assert settings.dense.credentials[0].provider == provider
        assert settings.dense.credentials[0].api_key == "draft"
        assert settings.max_retries == 0
        return instance

    monkeypatch.setattr(
        "openviking_cli.utils.config.embedding_config.EmbeddingConfig.get_embedder", build
    )
    assert (
        await probe_model(
            "embedding",
            {"provider": provider, "api_key": "draft"},
            {"model": "test-model", "dimension": 1},
        )
    )["ok"]
    instance.embed_async.assert_awaited_once_with("connection test")
    instance.close.assert_called_once()


async def test_keyless_embedding_uses_single_server_adapter(monkeypatch):
    instance = Mock(
        embed_async=AsyncMock(return_value=Mock(dense_vector=[0.1], sparse_vector=None))
    )

    def build(settings):
        assert len(settings.dense.credentials) == 1
        assert settings.dense.credentials[0].api_base == "http://localhost:8080/v1"
        return instance

    monkeypatch.setattr(
        "openviking_cli.utils.config.embedding_config.EmbeddingConfig.get_embedder", build
    )
    result = await probe_model(
        "embedding",
        {"provider": "openai", "api_base": "http://localhost:8080/v1"},
        {"model": "local", "dimension": 1},
    )
    assert result["ok"]
    instance.close.assert_called_once()


async def test_unknown_provider_does_not_initialize_model(vlm):
    assert await probe_model("vlm", {"provider": "unknown"}, {}) == {
        "ok": False,
        "code": "unsupported",
    }
    vlm.get_completion_async.assert_not_awaited()


async def test_empty_completion_is_not_reported_as_connected(vlm):
    vlm.get_completion_async.return_value = ""
    result = await probe_model("vlm", {"provider": "openai", "api_key": "draft"}, {"model": "o3"})
    assert result == {"ok": False, "code": "invalidResponse"}
    vlm.close.assert_called_once()


async def test_cancelled_probe_releases_instance(vlm):
    vlm.get_completion_async.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await probe_model("vlm", {"provider": "openai", "api_key": "draft"}, {"model": "o3"})
    vlm.close.assert_called_once()
