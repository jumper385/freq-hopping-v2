"""
Tests for app.py pure logic (det_msg / gen_msg).
No hardware or SDR required.
"""
import uuid
import pytest

from app import det_msg, gen_msg


@pytest.fixture
def dev_id():
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


class TestGenMsg:
    def test_includes_dev_id(self, dev_id):
        out = gen_msg("hello", dev_id)
        assert str(dev_id) in out

    def test_includes_message(self, dev_id):
        out = gen_msg("hello world", dev_id)
        assert "hello world" in out

    def test_format(self, dev_id):
        out = gen_msg("ping", dev_id)
        assert out == f"{dev_id} | ping"


class TestDetMsg:
    def test_own_message_filtered(self, dev_id):
        msg = gen_msg("test", dev_id)
        assert det_msg(msg, dev_id) is None

    def test_foreign_message_extracted(self, dev_id):
        other_id = uuid.uuid4()
        msg = gen_msg("hello", other_id)
        result = det_msg(msg, dev_id)
        assert result == "hello"

    def test_strips_whitespace(self, dev_id):
        other_id = uuid.uuid4()
        msg = f"{other_id} |   padded   "
        result = det_msg(msg, dev_id)
        assert result == "padded"

    def test_empty_message_part(self, dev_id):
        other_id = uuid.uuid4()
        msg = f"{other_id} | "
        result = det_msg(msg, dev_id)
        assert result == ""
