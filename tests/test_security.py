"""Tests for security utilities."""

import pytest
from vantinel_sdk.security import (
    hmac_sign,
    validate_collector_url,
    generate_nonce,
    redact_api_key,
    secure_zero,
)


class TestHmacSign:
    def test_returns_hex_string(self):
        sig = hmac_sign("my-key", 1700000000, '{"event": "test"}')
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        sig1 = hmac_sign("key", 100, "body")
        sig2 = hmac_sign("key", 100, "body")
        assert sig1 == sig2

    def test_different_keys_produce_different_signatures(self):
        sig1 = hmac_sign("key-a", 100, "body")
        sig2 = hmac_sign("key-b", 100, "body")
        assert sig1 != sig2

    def test_different_timestamps_produce_different_signatures(self):
        sig1 = hmac_sign("key", 100, "body")
        sig2 = hmac_sign("key", 200, "body")
        assert sig1 != sig2

    def test_different_bodies_produce_different_signatures(self):
        sig1 = hmac_sign("key", 100, '{"a": 1}')
        sig2 = hmac_sign("key", 100, '{"a": 2}')
        assert sig1 != sig2


class TestValidateCollectorUrl:
    def test_allows_https(self):
        assert validate_collector_url("https://collector.vantinel.com") == "https://collector.vantinel.com"

    def test_allows_localhost_http(self):
        assert validate_collector_url("http://localhost:8000") == "http://localhost:8000"

    def test_allows_127_0_0_1_http(self):
        assert validate_collector_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"

    def test_allows_0_0_0_0_http(self):
        assert validate_collector_url("http://0.0.0.0:8000") == "http://0.0.0.0:8000"

    def test_allows_ipv6_loopback_http(self):
        assert validate_collector_url("http://[::1]:8000") == "http://[::1]:8000"

    def test_allows_private_10_x(self):
        assert validate_collector_url("http://10.0.0.5:8000") == "http://10.0.0.5:8000"

    def test_allows_private_192_168(self):
        assert validate_collector_url("http://192.168.1.100:8000") == "http://192.168.1.100:8000"

    def test_allows_private_172_16_to_31(self):
        for i in range(16, 32):
            url = f"http://172.{i}.0.1:8000"
            assert validate_collector_url(url) == url

    def test_rejects_public_http(self):
        with pytest.raises(ValueError, match="HTTPS"):
            validate_collector_url("http://collector.example.com")

    def test_rejects_public_http_ip(self):
        with pytest.raises(ValueError, match="HTTPS"):
            validate_collector_url("http://8.8.8.8:8000")


class TestGenerateNonce:
    def test_returns_hex_string(self):
        nonce = generate_nonce()
        assert isinstance(nonce, str)
        assert len(nonce) == 32  # 16 bytes = 32 hex chars

    def test_hex_characters_only(self):
        nonce = generate_nonce()
        assert all(c in "0123456789abcdef" for c in nonce)

    def test_unique(self):
        nonces = {generate_nonce() for _ in range(100)}
        assert len(nonces) == 100


class TestRedactApiKey:
    def test_redacts_long_key(self):
        result = redact_api_key("vntg_liv_abcdef1234567890")
        assert result == "vntg****7890"

    def test_short_key_fully_redacted(self):
        assert redact_api_key("short") == "****"
        assert redact_api_key("12345678") == "****"

    def test_exactly_9_chars_shows_prefix_suffix(self):
        result = redact_api_key("123456789")
        assert result == "1234****6789"

    def test_empty_string(self):
        assert redact_api_key("") == "****"


class TestSecureZero:
    def test_does_not_raise_on_empty_string(self):
        # Should be a no-op
        secure_zero("")

    def test_does_not_raise_on_normal_string(self):
        # We can't easily verify the memory is zeroed, but it shouldn't crash
        s = "a" * 100  # Avoid interned strings
        secure_zero(s)

    def test_does_not_raise_on_long_string(self):
        s = "x" * 10000
        secure_zero(s)
