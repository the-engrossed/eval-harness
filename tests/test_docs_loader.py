from pathlib import Path

import pytest

from auditlab.docs.loader import load_html_file, parse_html, sha256_hex


def test_pages_split_on_section_and_tables_are_linearized() -> None:
    html = """
    <html><body>
      <section class="page" data-page-no="1">
        <h1>Invoice</h1>
        <table>
          <tr><th>Item</th><th>Amount</th></tr>
          <tr><td>Pallet racking</td><td>4,500.00</td></tr>
        </table>
      </section>
      <section class="page" data-page-no="2">
        <p>Goods shipped on 2025-11-03.</p>
      </section>
    </body></html>
    """
    parsed = parse_html(html, source_uri="memory://fixture")
    assert parsed.page_count == 2
    assert "Pallet racking | 4,500.00" in parsed.text_for(1)
    assert "Goods shipped on 2025-11-03." in parsed.text_for(2)


def test_missing_page_sections_fail_loudly() -> None:
    with pytest.raises(ValueError, match="no <section"):
        parse_html("<html><p>no pages</p></html>", source_uri="memory://bad")


def test_non_contiguous_pages_fail_loudly() -> None:
    html = """
    <section class="page" data-page-no="1"><p>one</p></section>
    <section class="page" data-page-no="3"><p>three</p></section>
    """
    with pytest.raises(ValueError, match="contiguous"):
        parse_html(html, source_uri="memory://gap")


def test_sha256_is_stable_and_file_load_round_trips(tmp_path: Path) -> None:
    html = '<section class="page" data-page-no="1"><p>Harbor Lane Supply Co.</p></section>'
    path = tmp_path / "doc.html"
    path.write_text(html, encoding="utf-8")
    loaded = load_html_file(path)
    assert loaded.sha256 == sha256_hex(html.encode("utf-8"))
    assert loaded.pages[0].text == "Harbor Lane Supply Co."
    again = load_html_file(path)
    assert again.sha256 == loaded.sha256
