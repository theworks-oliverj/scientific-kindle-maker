"""
Tests for volume splitting in s10_epub_assembly.

Amazon's Send to Kindle converter rejects a book whose total inline <svg> count
is too high, with a bare E999 and no diagnostic. Established by bisection across
15 uploads: 1054 elements converts, 1342 does not. Kindle Previewer does NOT
reproduce it, so the only protection is staying under budget.

Every other candidate metric was contradicted by a direct pass/fail inversion
(compressed size, uncompressed bytes, XHTML file count, svg-bearing file count,
PNG count), so the split has to be driven by equation count specifically.
"""
import zipfile

from kindle_math_converter.stages.s10_epub_assembly import (
    KINDLE_MAX_EQUATIONS_PER_VOLUME,
    _equation_count,
    _partition_into_volumes,
)


def _ch(title: "str | None", n_svg: int) -> "tuple[str | None, str]":
    """One (title, xhtml) chapter carrying n_svg inline equations."""
    return (title, "<html><body>" + "<svg></svg>" * n_svg + "</body></html>")


def test_equation_count_counts_svg_elements():
    assert _equation_count("<svg></svg><svg></svg>") == 2
    assert _equation_count("<p>no maths here</p>") == 0


def test_a_book_under_budget_stays_one_volume():
    chapters = [_ch("A", 100), _ch("B", 100)]
    assert _partition_into_volumes(chapters, 900) == [[0, 1]]


def test_a_book_over_budget_is_split():
    # 400+400 packs into one volume; adding the third would reach 1200 > 900
    chapters = [_ch("A", 400), _ch("B", 400), _ch("C", 400)]
    vols = _partition_into_volumes(chapters, 900)
    assert vols == [[0, 1], [2]]


def test_chapters_that_cannot_share_a_volume_are_not_packed():
    # 500+500 exceeds 900, so each takes its own volume
    chapters = [_ch("A", 500), _ch("B", 500), _ch("C", 500)]
    assert _partition_into_volumes(chapters, 900) == [[0], [1], [2]]


def test_every_volume_stays_within_budget():
    chapters = [_ch(f"C{i}", 300) for i in range(10)]
    vols = _partition_into_volumes(chapters, 900)
    for v in vols:
        total = sum(_equation_count(chapters[i][1]) for i in v)
        assert total <= 900


def test_continuation_files_are_never_split_from_their_chapter():
    # title=None marks a MAX_CHAPTER_BYTES continuation; cutting before one
    # would strand content in a different volume from its heading
    chapters = [_ch("A", 800), (None, "<svg></svg>" * 800), _ch("B", 100)]
    vols = _partition_into_volumes(chapters, 900)
    assert [0, 1] == vols[0], "continuation must stay with its titled chapter"


def test_a_single_oversized_chapter_gets_its_own_volume():
    chapters = [_ch("A", 100), _ch("HUGE", 2000), _ch("B", 100)]
    vols = _partition_into_volumes(chapters, 900)
    assert [1] in vols, "an indivisible over-budget chapter is isolated"


def test_no_chapter_is_lost_or_duplicated():
    chapters = [_ch(f"C{i}", 250) for i in range(17)]
    vols = _partition_into_volumes(chapters, 900)
    flat = [i for v in vols for i in v]
    assert flat == sorted(flat)
    assert flat == list(range(17))


def test_budget_of_none_or_zero_disables_splitting():
    chapters = [_ch(f"C{i}", 1000) for i in range(5)]
    assert _partition_into_volumes(chapters, 0) == [[0, 1, 2, 3, 4]]


def test_default_budget_sits_below_the_largest_observed_pass():
    # 1054 equations converted at Amazon; 1342 did not. The default must leave
    # headroom under 1054 rather than creep up towards the failing value.
    assert KINDLE_MAX_EQUATIONS_PER_VOLUME < 1054


def _run_assembly(tmp_path, chapters, budget, monkeypatch):
    """Drive s10.run() with a stubbed chapter builder, no epubcheck."""
    from kindle_math_converter.observability.event_bus import EventBus
    from kindle_math_converter.stages import s10_epub_assembly as s10
    from kindle_math_converter.models.document import Document, DocumentMetadata
    from kindle_math_converter.models.enums import ColumnLayout, SourceType

    monkeypatch.setattr(s10, "_document_to_chapters",
                        lambda *a, **k: [(t, s10.XHTML_TEMPLATE.format(title=t or "c", body=b))
                                         for t, b in chapters])
    doc = Document(metadata=DocumentMetadata(
        source_path="book.pdf", source_type=SourceType.LATEX_PDF, title="Book",
        author="A", page_count=1, has_math_fonts=True, is_scanned=False,
        column_layout=ColumnLayout.SINGLE))
    out = tmp_path / "book_assembled.epub"
    path, res = s10.run(doc, out, EventBus(), epubcheck_enabled=False,
                        max_equations_per_volume=budget)
    return path, res


def test_run_writes_one_file_when_under_budget(tmp_path, monkeypatch):
    chapters = [("A", "<svg></svg>" * 10)]
    path, res = _run_assembly(tmp_path, chapters, 900, monkeypatch)
    assert res.ok
    assert path.name == "book_assembled.epub"
    assert res.metrics["volumes"] == 1
    assert not list(tmp_path.glob("*_vol*.epub"))


def test_run_writes_separate_valid_volumes_when_over_budget(tmp_path, monkeypatch):
    chapters = [(f"C{i}", "<svg></svg>" * 400) for i in range(4)]
    path, res = _run_assembly(tmp_path, chapters, 900, monkeypatch)
    assert res.ok
    assert res.metrics["volumes"] == 2
    assert res.metrics["equations"] == 1600
    vols = sorted(tmp_path.glob("book_assembled_vol*.epub"))
    assert len(vols) == 2
    assert path == vols[0]

    seen_chapters = 0
    for v in vols:
        with zipfile.ZipFile(v) as z:
            names = z.namelist()
            assert names[0] == "mimetype"
            assert z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
            for required in ("META-INF/container.xml", "OEBPS/content.opf",
                            "OEBPS/nav.xhtml", "OEBPS/toc.ncx"):
                assert required in names, f"{required} missing from {v.name}"
            chs = [n for n in names if n.startswith("OEBPS/content/")]
            seen_chapters += len(chs)
            # each volume renumbers its own chapters from 001
            assert "OEBPS/content/chapter_001.xhtml" in names
            opf = z.read("OEBPS/content.opf").decode()
            for ch in chs:
                assert ch.split("OEBPS/")[1] in opf, "chapter missing from manifest"
    assert seen_chapters == 4, "chapters lost or duplicated across volumes"


def test_volumes_get_distinct_identifiers_and_titles(tmp_path, monkeypatch):
    chapters = [(f"C{i}", "<svg></svg>" * 400) for i in range(4)]
    _run_assembly(tmp_path, chapters, 900, monkeypatch)
    uids, titles = set(), set()
    for v in sorted(tmp_path.glob("book_assembled_vol*.epub")):
        with zipfile.ZipFile(v) as z:
            opf = z.read("OEBPS/content.opf").decode()
        uids.add(opf.split('id="uid">')[1].split("<")[0])
        titles.add(opf.split("<dc:title>")[1].split("</dc:title>")[0])
    # colliding uids would let Kindle dedup one volume against another
    assert len(uids) == 2, "volumes must not share a dc:identifier"
    assert len(titles) == 2, "volumes must be distinguishable in the library"
    assert all("Volume" in t for t in titles)
