import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bropilot.utils import codex_backend as backend
from bropilot.utils.llm import LlmClient, LlmError, encode_image


def test_chatgpt_transport_text_image_and_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, 'chatgpt_ready', lambda: (True, 'ok'))
    monkeypatch.setattr(backend, 'executable', lambda: '/codex')
    monkeypatch.setenv('OPENAI_API_KEY', 'do-not-use')
    monkeypatch.setenv('CODEX_API_KEY', 'do-not-use')
    image = tmp_path / 'board.png'
    image.write_bytes(b'image-bytes')

    def run(cmd, **kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env']
        assert 'CODEX_API_KEY' not in kwargs['env']
        assert 'forced_login_method="chatgpt"' in cmd
        assert cmd[cmd.index('--sandbox') + 1] == 'read-only'
        attached = Path(cmd[cmd.index('--image') + 1])
        assert attached.read_bytes() == b'image-bytes'
        assert 'lecture task' in kwargs['input']
        assert 'system instruction' in kwargs['input']
        Path(cmd[cmd.index('-o') + 1]).write_text('{"ok":true}')
        event = {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 5}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(event), stderr='')

    monkeypatch.setattr(backend.subprocess, 'run', run)
    client = LlmClient('codex')
    assert client.complete_json([{'type': 'text', 'text': 'lecture task'}, encode_image(image)],
                                system='system instruction') == {'ok': True}
    assert client.report() == {'model': 'codex', 'calls': 1, 'input_tokens': 12, 'output_tokens': 5}


def test_no_api_fallback_when_chatgpt_not_logged_in(monkeypatch):
    monkeypatch.setattr(backend, 'chatgpt_ready', lambda: (False, 'Log in with ChatGPT'))
    with pytest.raises(LlmError, match='Log in with ChatGPT'):
        LlmClient('codex')


def test_failed_turn_is_not_counted_as_success(monkeypatch):
    monkeypatch.setattr(backend, 'chatgpt_ready', lambda: (True, 'ok'))
    monkeypatch.setattr(backend, 'executable', lambda: '/codex')
    monkeypatch.setattr(backend.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout='{"type":"turn.failed","error":{"message":"limit reached"}}', stderr=''))
    client = LlmClient('codex')
    with pytest.raises(LlmError, match='limit reached'):
        client.complete('lecture task')
    assert client.report()['calls'] == 0
