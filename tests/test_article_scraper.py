"""Tests for src/utils/article_scraper.py — security and validation."""

import ipaddress
import socket
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from utils.article_scraper import (
    _validate_url,
    _check_private_ip,
    scrape_article,
    _MAX_RESPONSE_BYTES,
)


class TestValidateUrl:
    def test_http_url_is_valid(self):
        _validate_url("http://example.com/article")  # must not raise

    def test_https_url_is_valid(self):
        _validate_url("https://example.com/article")  # must not raise

    def test_ftp_raises(self):
        with pytest.raises(ValueError, match="http"):
            _validate_url("ftp://example.com/file")

    def test_javascript_raises(self):
        with pytest.raises(ValueError):
            _validate_url("javascript:alert(1)")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_url("")

    def test_no_netloc_raises(self):
        with pytest.raises(ValueError):
            _validate_url("http://")

    def test_data_uri_raises(self):
        with pytest.raises(ValueError):
            _validate_url("data:text/html,<h1>hi</h1>")


class TestCheckPrivateIp:
    def _make_addr_info(self, ip_str: str):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip_str, 80))]

    def test_loopback_is_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("127.0.0.1")):
            with pytest.raises(ValueError, match="blocked"):
                _check_private_ip("localhost")

    def test_private_class_a_is_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("10.0.0.1")):
            with pytest.raises(ValueError, match="blocked"):
                _check_private_ip("internal.corp")

    def test_private_class_c_is_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("192.168.1.100")):
            with pytest.raises(ValueError, match="blocked"):
                _check_private_ip("router.local")

    def test_cloud_metadata_ip_is_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("169.254.169.254")):
            with pytest.raises(ValueError, match="blocked"):
                _check_private_ip("metadata.internal")

    def test_public_ip_passes(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("8.8.8.8")):
            _check_private_ip("dns.google")  # must not raise

    def test_another_public_ip_passes(self):
        with patch("socket.getaddrinfo", return_value=self._make_addr_info("1.1.1.1")):
            _check_private_ip("one.one.one.one")  # must not raise

    def test_dns_failure_raises(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            with pytest.raises(ValueError, match="DNS resolution failed"):
                _check_private_ip("no-such-host.invalid")


class TestScrapeArticle:
    def _make_mock_response(self, html: bytes, content_type: str = "text/html; charset=utf-8",
                            charset: str = "utf-8"):
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = content_type
        mock_resp.headers.get_content_charset.return_value = charset
        mock_resp.read.return_value = html
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_successful_scrape(self, mock_urlopen, mock_check_ip):
        html = b"""<html><head><title>Test Article</title></head><body>
        <p>Scientists confirmed that the experiment produced results in 2023.
        The study was conducted over three years and involved 500 participants.</p>
        </body></html>"""
        mock_urlopen.return_value = self._make_mock_response(html)

        result = scrape_article("https://example.com/article")
        assert result.title == "Test Article"
        assert len(result.text) > 0
        assert result.url == "https://example.com/article"

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_oversized_response_raises(self, mock_urlopen, mock_check_ip):
        # Return more than 5MB
        mock_resp = self._make_mock_response(b"")
        mock_resp.read.return_value = b"x" * (_MAX_RESPONSE_BYTES + 2)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(ValueError, match="exceeds maximum size"):
            scrape_article("https://example.com/big")

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_unsupported_content_type_raises(self, mock_urlopen, mock_check_ip):
        mock_urlopen.return_value = self._make_mock_response(
            b"binary data", content_type="application/pdf"
        )
        with pytest.raises(ValueError, match="Unsupported content type"):
            scrape_article("https://example.com/file.pdf")

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_http_error_raises(self, mock_urlopen, mock_check_ip):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="https://example.com", code=404,
            msg="Not Found", hdrs=None, fp=None
        )
        with pytest.raises(ValueError, match="HTTP 404"):
            scrape_article("https://example.com/missing")

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_url_error_raises(self, mock_urlopen, mock_check_ip):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("connection refused")
        with pytest.raises(ValueError, match="Failed to fetch"):
            scrape_article("https://example.com/unreachable")

    def test_invalid_scheme_raises_before_network(self):
        with pytest.raises(ValueError):
            scrape_article("ftp://example.com/file")

    def test_ssrf_private_ip_blocked(self):
        """_check_private_ip is called before urlopen — network never reached for private IPs."""
        with patch("utils.article_scraper._check_private_ip",
                   side_effect=ValueError("Access to private/reserved IP addresses is blocked: 127.0.0.1")):
            with pytest.raises(ValueError, match="blocked"):
                scrape_article("https://internal.corp/data")

    @patch("utils.article_scraper._check_private_ip")
    @patch("utils.article_scraper.urlopen")
    def test_too_short_text_raises(self, mock_urlopen, mock_check_ip):
        html = b"<html><body><p>Short.</p></body></html>"
        mock_urlopen.return_value = self._make_mock_response(html)
        with pytest.raises(ValueError, match="too short"):
            scrape_article("https://example.com/tiny")
