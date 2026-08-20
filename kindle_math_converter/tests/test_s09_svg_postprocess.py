# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Oliver Jandette

"""
Regression coverage for the dvisvgm orphaned-glyph-reference detector.

Background: Stage 8A's per-equation gate (find_orphaned_use_refs, called
from s08a_svg_render._render_region) is meant to catch dvisvgm's occasional
"references a glyph it never defined" bug and route the equation to a raster
fallback instead of shipping a silently broken SVG. On a 2205-equation book,
9 equations got past that gate anyway and were only caught by Stage 10's
whole-document check, after the entire book had already been assembled --
turning a graceful per-equation fallback into a fatal, whole-book failure
over 0.4% of equations.

Root cause: the two checks used different definitions of "defined". Stage
8A's original regex counted an id="..." on ANY element as a definition;
Stage 10's counted only <path id="..."> id="..."> as one. dvisvgm's broken
render occasionally emits the id on a non-<path> wrapper element instead of
dropping it entirely, which the looser check reads as "defined" and the
stricter one correctly reads as orphaned. Stage 8A's check must always be at
least as strict as Stage 10's -- it is the one meant to catch this early.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kindle_math_converter.stages.s09_svg_postprocess import find_orphaned_use_refs


class TestFindOrphanedUseRefs(unittest.TestCase):
    def test_clean_svg_has_no_orphans(self):
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg'>"
            "<defs><path id='g1-98' d='M0 0'/></defs>"
            "<g id='page1'><use x='0' y='0' xlink:href='#g1-98'/></g>"
            "</svg>"
        )
        self.assertEqual(find_orphaned_use_refs(svg), set())

    def test_fully_missing_glyph_is_orphaned(self):
        # The original (2026-08-14) incident shape: the id never appears
        # anywhere in the SVG, not even on a non-path element.
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg'>"
            "<defs><path id='g1-98' d='M0 0'/></defs>"
            "<g id='page1'>"
            "<use x='0' y='0' xlink:href='#g1-98'/>"
            "<use x='5' y='0' xlink:href='#g1-82'/>"
            "</g>"
            "</svg>"
        )
        self.assertEqual(find_orphaned_use_refs(svg), {"g1-82"})

    def test_glyph_id_on_non_path_wrapper_is_still_orphaned(self):
        # The 2026-08-19 incident shape: dvisvgm's broken glyph slot carries
        # its id on a non-<path> element (observed: a <g>) instead of
        # dropping it. No <path id="g1-126"> exists anywhere, so the glyph
        # itself was never actually drawn -- this must be flagged even
        # though the string "g1-126" appears as *some* element's id.
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg'>"
            "<defs>"
            "<path id='g1-98' d='M0 0'/>"
            "<g id='g1-126'><use xlink:href='#g1-98'/></g>"
            "</defs>"
            "<g id='page1'>"
            "<use x='0' y='0' xlink:href='#g1-98'/>"
            "<use x='5' y='0' xlink:href='#g1-126'/>"
            "</g>"
            "</svg>"
        )
        self.assertEqual(find_orphaned_use_refs(svg), {"g1-126"})

    def test_matches_raw_dvisvgm_single_quote_xlink_form(self):
        # find_orphaned_use_refs must work on dvisvgm's raw output too
        # (s08a_svg_render calls it before any quote/xlink normalization).
        svg = (
            "<svg version='1.1' xmlns='http://www.w3.org/2000/svg' "
            "xmlns:xlink='http://www.w3.org/1999/xlink'>"
            "<defs><path id='g1-98' d='M0 0'/></defs>"
            "<g id='page1'><use xlink:href='#g1-126'/></g>"
            "</svg>"
        )
        self.assertEqual(find_orphaned_use_refs(svg), {"g1-126"})


class TestS10DelegatesToSharedPrimitive(unittest.TestCase):
    """Stage 10's whole-document check must never diverge from Stage 8A's
    per-equation gate -- that drift is exactly what let 9 equations ship
    with a genuinely undefined glyph. Verify it delegates rather than
    keeping its own regex."""

    def test_svgs_referencing_outside_themselves_flags_wrapper_case(self):
        from kindle_math_converter.stages.s10_epub_assembly import (
            _svgs_referencing_outside_themselves,
        )

        xhtml = (
            "<html><body>"
            "<svg xmlns='http://www.w3.org/2000/svg'>"
            "<defs>"
            "<path id='eq1__g1-98' d='M0 0'/>"
            "<g id='eq1__g1-126'><use href='#eq1__g1-98'/></g>"
            "</defs>"
            "<g id='page1'>"
            "<use x='0' y='0' href='#eq1__g1-98'/>"
            "<use x='5' y='0' href='#eq1__g1-126'/>"
            "</g>"
            "</svg>"
            "</body></html>"
        )
        self.assertEqual(
            _svgs_referencing_outside_themselves(xhtml), ["eq1__g1-126"]
        )


if __name__ == "__main__":
    unittest.main()
