"""URL minimization, canonical keys and offline-domain privacy regressions."""
from src.data.url import normalize_url


class TestURLPrivacy:
    def test_query_and_fragment_are_destroyed(self):
        """Privacy regression: a token in the query must vanish."""
        n = normalize_url("https://site.example.com/a?session=SUPERSECRET123#frag")
        blob = f"{n.host}{n.path}{n.canonical}{n.feature_text}{n.registrable_domain}{n.raw}"
        assert "SUPERSECRET123" not in blob
        assert "session" not in blob
        assert "frag" not in blob

    def test_userinfo_and_port_are_destroyed(self):
        n = normalize_url("http://alice:hunter2@site.example.org:8443/x")
        blob = f"{n.host}{n.path}{n.canonical}{n.raw}"
        assert "alice" not in blob and "hunter2" not in blob and "8443" not in blob

    def test_registrable_domain_handles_multipart_suffix(self):
        assert normalize_url("https://a.b.example.co.uk/x").registrable_domain == "example.co.uk"

    def test_invalid_urls_are_rejected_not_guessed(self):
        for bad in ["", "   ", "ftp://x.example.com", "http://localhost/x", "not a url"]:
            assert normalize_url(bad).is_valid is False

    def test_invalid_url_raw_is_still_scrubbed(self):
        n = normalize_url("ftp://alice:hunter2@x.example.com/a?token=SECRET#frag")
        assert n.is_valid is False
        assert "SECRET" not in n.raw
        assert "hunter2" not in n.raw
        assert "frag" not in n.raw

    def test_bare_host_gets_scheme_assumed(self):
        n = normalize_url("example.com/path")
        assert n.is_valid and n.scheme == "http" and n.registrable_domain == "example.com"

    def test_canonical_ignores_scheme_and_trailing_slash(self):
        a = normalize_url("https://x.example.com/a/")
        b = normalize_url("http://x.example.com/a")
        assert a.canonical == b.canonical

    def test_normalize_never_raises(self):
        for weird in [None, 12345, "http://[bad", "http://" + "a" * 5000, "://"]:
            normalize_url(weird)


def test_bare_url_userinfo_is_scrubbed():
    n1 = normalize_url("alice:hunter2@example.com/path")
    assert n1.is_valid is True
    assert "hunter2" not in n1.raw
    assert "alice" not in n1.raw

    n2 = normalize_url("alice:hunter2@localhost")
    assert n2.is_valid is False
    assert "hunter2" not in n2.raw
    assert "alice" not in n2.raw
