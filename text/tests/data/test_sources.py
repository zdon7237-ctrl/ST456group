from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.download_gutenberg import download_source
from novel_continuation.data_sources import SHERLOCK_SOURCES


def test_sherlock_sources_have_titles_book_ids_and_https_urls():
    assert SHERLOCK_SOURCES
    for item in SHERLOCK_SOURCES:
        assert item["book_id"]
        assert item["title"]
        assert item["url"].startswith("https://")


def test_download_source_skips_existing_file(tmp_path, monkeypatch, capsys):
    target_path = tmp_path / "1661.txt"
    target_path.write_text("already-downloaded", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network should not be used when file already exists")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)

    status = download_source(
        title="The Adventures of Sherlock Holmes",
        url="https://example.com/book.txt",
        target_path=target_path,
    )

    captured = capsys.readouterr()
    assert status == "skipped"
    assert "Skipping existing file" in captured.out


def test_download_source_uses_timeout_and_writes_bytes(tmp_path, monkeypatch):
    target_path = tmp_path / "834.txt"
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"sherlock"

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    status = download_source(
        title="The Memoirs of Sherlock Holmes",
        url="https://example.com/memoirs.txt",
        target_path=target_path,
        timeout=12.5,
    )

    assert status == "downloaded"
    assert observed == {
        "url": "https://example.com/memoirs.txt",
        "timeout": 12.5,
    }
    assert target_path.read_bytes() == b"sherlock"
