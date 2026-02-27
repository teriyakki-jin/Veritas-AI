"""Lightweight article scraper for URL-based analysis.

This module intentionally avoids heavy dependencies so it can run in
restricted environments. It extracts title and paragraph text from HTML.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB


def _check_private_ip(hostname: str) -> None:
    """Resolve hostname and block private/loopback/link-local/reserved IPs (SSRF prevention)."""
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for {hostname!r}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Access to private/reserved IP addresses is blocked: {ip_str}")


@dataclass
class ScrapedArticle:
    url: str
    title: str
    text: str


def _strip_tags(raw_html: str) -> str:
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", no_script, flags=re.IGNORECASE)
    no_tags = re.sub(r"<[^>]+>", " ", no_style)
    unescaped = html.unescape(no_tags)
    cleaned = re.sub(r"\s+", " ", unescaped).strip()
    return cleaned


def _extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw_html, flags=re.IGNORECASE)
    if not match:
        return ""
    title = _strip_tags(match.group(1))
    return title[:300]


def _extract_paragraph_text(raw_html: str) -> str:
    paragraphs = re.findall(r"<p[^>]*>([\s\S]*?)</p>", raw_html, flags=re.IGNORECASE)
    blocks = []

    for paragraph in paragraphs:
        text = _strip_tags(paragraph)
        # Filter out boilerplate-like tiny blocks.
        if len(text) >= 60:
            blocks.append(text)

    if not blocks:
        # Fallback: use stripped full document text.
        return _strip_tags(raw_html)

    # Keep enough context, but cap size for inference speed.
    joined = "\n".join(blocks[:40])
    return joined[:15000]


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must start with http:// or https://")


def scrape_article(url: str, timeout_sec: int = 15, user_agent: Optional[str] = None) -> ScrapedArticle:
    """Fetch URL and extract title + article-like text."""
    clean_url = url.strip()
    _validate_url(clean_url)

    parsed = urlparse(clean_url)
    _check_private_ip(parsed.hostname)

    headers = {
        "User-Agent": user_agent
        or "Mozilla/5.0 (compatible; VeritasAI/1.0; +https://github.com/teriyakki-jin/Veritas-AI)"
    }

    request = Request(clean_url, headers=headers)

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise ValueError(f"Unsupported content type: {content_type}")

            raw_bytes = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw_bytes) > _MAX_RESPONSE_BYTES:
                raise ValueError(
                    f"Response exceeds maximum size of {_MAX_RESPONSE_BYTES // (1024 * 1024)} MB"
                )
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = raw_bytes.decode(charset, errors="replace")
    except HTTPError as exc:
        raise ValueError(f"Failed to fetch article: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Failed to fetch article: {exc.reason}") from exc

    title = _extract_title(raw_html)
    text = _extract_paragraph_text(raw_html)

    if len(text) < 80:
        raise ValueError("Article text is too short or could not be extracted")

    return ScrapedArticle(url=clean_url, title=title, text=text)
