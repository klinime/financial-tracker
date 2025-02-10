import re
from itertools import accumulate
from logging import getLogger
from sys import maxsize
from typing import Any

import fitz
import numpy as np

point_threshold = 3.0  # 3 points is 1/24 inches


def text_block_equivalent(
    text_block1: dict[str, Any], text_block2: dict[str, Any]
) -> bool:
    rect1 = fitz.Rect(text_block1["bbox"])
    rect2 = fitz.Rect(text_block2["bbox"])
    overlap = rect1 & rect2
    rect1_area = rect1.width * rect1.height
    rect2_area = rect2.width * rect2.height
    overlap_area = overlap.width * overlap.height
    # sufficient area must overlap to be considered equivalent spatially
    if (overlap_area / rect1_area) < 0.5 or (overlap_area / rect2_area) < 0.5:
        return False
    # number of lines must be the same
    if len(text_block1["lines"]) != len(text_block2["lines"]):
        return False
    # line text must be the same, barring some minor formatting differences
    for line1, line2 in zip(text_block1["lines"], text_block2["lines"]):
        if re.sub(r"\d+ of (\d+)", r"of \1", line1["spans"][0]["text"]) != re.sub(
            r"\d+ of (\d+)", r"of \1", line2["spans"][0]["text"]
        ):
            return False
    return True


def sorted_header_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # sort by bottom of the block because excluding header goes from bottom to top
    return sorted(text_blocks, key=lambda block: block["bbox"][3])


def sorted_footer_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # sort by top of the block because excluding footer goes from top to bottom
    return sorted(text_blocks, key=lambda block: block["bbox"][1], reverse=True)


def pdf_margin_offsets(doc: fitz.Document) -> tuple[int, int]:
    logger = getLogger(__name__)

    # reference text blocks assumes maximally 2 pages before the start of the content page
    # page 0: cover, page 1: terms or summary, page 2: content
    ref_page_count = min(doc.page_count, 3)
    ref_page_blocks = [
        [
            block
            for block in doc[page_num].get_text("dict")["blocks"]
            if block["type"] == 0
        ]
        for page_num in range(ref_page_count)
    ]
    header_offset = maxsize
    footer_offset = 0
    for page_num in range(1, doc.page_count):
        # dedplicate equivalence checks between the reference pages
        cur_ref_blocks = [
            page_block
            for ref_page_num, page_block in enumerate(ref_page_blocks)
            if page_num > ref_page_num
        ]
        page_text_blocks = [
            block
            for block in doc[page_num].get_text("dict")["blocks"]
            if block["type"] == 0
        ]
        header_bottom = 0
        footer_top = doc[page_num].rect.height
        header_line_count = 0
        footer_line_count = 0
        for ref_text_blocks in cur_ref_blocks:
            # check against reference header blocks for equivalent prefixes
            for ref_text_block, page_text_block in zip(
                sorted_header_blocks(ref_text_blocks),
                sorted_header_blocks(page_text_blocks),
            ):
                if text_block_equivalent(ref_text_block, page_text_block):
                    header_bottom = page_text_block["bbox"][3]
                    header_line_count += len(page_text_block["lines"])
                else:
                    break
            # check against reference footer blocks for equivalent suffixes
            for ref_text_block, page_text_block in zip(
                sorted_footer_blocks(ref_text_blocks),
                sorted_footer_blocks(page_text_blocks),
            ):
                if text_block_equivalent(ref_text_block, page_text_block):
                    footer_top = page_text_block["bbox"][1]
                    footer_line_count += len(page_text_block["lines"])
                else:
                    break
        # at least 2 lines of text to be considered the header or footer
        if header_bottom != 0 and header_line_count > 2:
            header_offset = min(header_offset, header_bottom)
        if footer_top != doc[page_num].rect.height and footer_line_count > 2:
            footer_offset = max(footer_offset, footer_top)
    if header_offset == maxsize:
        header_offset = 0
    if footer_offset == 0:
        footer_offset = doc[0].rect.height

    # debug info for texts removed via header / footer clipping
    logger.debug(f"{header_offset=} {footer_offset=}")
    for page_num in range(ref_page_count):
        header_clip = fitz.Rect(0, 0, doc[page_num].rect.width, header_offset)
        logger.debug(f"removed_header:\n{doc[page_num].get_text(clip=header_clip)}")
        footer_clip = fitz.Rect(
            0, footer_offset, doc[page_num].rect.width, doc[page_num].rect.height
        )
        logger.debug(f"removed_footer:\n{doc[page_num].get_text(clip=footer_clip)}")

    return header_offset, footer_offset


def is_two_column_page(page: fitz.Page) -> bool:
    text_blocks = [
        block for block in page.get_text("dict")["blocks"] if block["type"] == 0
    ]
    if not text_blocks:
        return False
    left_blocks = [
        block for block in text_blocks if block["bbox"][2] < page.rect.width / 2
    ]
    right_blocks = [
        block for block in text_blocks if block["bbox"][0] > page.rect.width / 2
    ]
    if not left_blocks or not right_blocks:
        return False
    left_boundary = float(
        np.quantile([block["bbox"][0] for block in left_blocks], 0.25)
    )
    right_boundary = float(
        np.quantile([block["bbox"][2] for block in right_blocks], 0.75)
    )
    center = (left_boundary + right_boundary) / 2
    # weighted distance to the center of the page based on the area of the block
    left_blocks_areas = [
        (block["bbox"][2] - block["bbox"][0]) * (block["bbox"][3] - block["bbox"][1])
        for block in left_blocks
    ]
    right_blocks_areas = [
        (block["bbox"][2] - block["bbox"][0]) * (block["bbox"][3] - block["bbox"][1])
        for block in right_blocks
    ]
    left_dist_weighted = sum(
        [
            (center - block["bbox"][2]) * area
            for block, area in zip(left_blocks, left_blocks_areas)
        ]
    ) / sum(left_blocks_areas)
    right_dist_weighted = sum(
        [
            (block["bbox"][0] - center) * area
            for block, area in zip(right_blocks, right_blocks_areas)
        ]
    ) / sum(right_blocks_areas)
    # arbitrary thresholds - too small = no clear separation between columns, too large = too sparse
    assert isinstance(left_dist_weighted, float)
    assert isinstance(right_dist_weighted, float)
    return (
        (left_dist_weighted > point_threshold)
        and (left_dist_weighted < point_threshold * 8)
        and (right_dist_weighted > point_threshold)
        and (right_dist_weighted < point_threshold * 8)
    )


def mask_sensitive_info(page: fitz.Page) -> fitz.Page:
    logger = getLogger(__name__)
    regexes = [
        re.compile(rf"{identifier}[:]? (\d+)")
        for identifier in [
            "Account Number",
            "Trace Number",
            "Member Number",
            "Employee Number",
        ]
    ]

    fitz.TOOLS.set_small_glyph_heights(True)
    # x0, y0, x1, y1, text, block_no, line_no, word_no = word
    words = page.get_text("words")
    text = " ".join([word[4] for word in words])
    results = [match.span(1) for regex in regexes for match in regex.finditer(text)]
    word_offsets = [i - 1 for i in accumulate([len(word[4]) + 1 for word in words])]
    for result in results:
        start, end = result
        redact_text = text[start:end]
        start_idx = next(
            (i for i in range(len(word_offsets)) if word_offsets[i] >= start),
            len(word_offsets),
        )
        end_idx = (
            next(
                (i for i in range(len(word_offsets)) if word_offsets[i] >= end),
                len(word_offsets),
            )
            + 1
        )
        # create the bounding box of the identified pii entity
        entity_bbox = fitz.Rect(words[start_idx][:4])
        for word in words[start_idx + 1 : end_idx]:
            entity_bbox |= fitz.Rect(word[:4])
        actual_text = page.get_text(clip=entity_bbox).strip()
        page.add_redact_annot(entity_bbox, fill=(1, 1, 1))
        page.apply_redactions(text=1)
        logger.debug(f"expect to redact: {redact_text}")
        logger.debug(f"actually redacted: {actual_text}")
        if redact_text != actual_text:
            logger.warning(
                f"Mismatch between expected and actual text: {redact_text} != {actual_text}"
            )

    fitz.TOOLS.set_small_glyph_heights(False)
    return page


def extract_pdf_pages(doc: fitz.Document) -> list[fitz.Page]:
    logger = getLogger(__name__)

    pages: list[fitz.Page] = []
    header_offset, footer_offset = pdf_margin_offsets(doc)
    for page in doc:
        if (header_offset > 0) or (footer_offset > 0):
            page.set_cropbox(
                fitz.Rect(0, header_offset, page.rect.width, footer_offset)
            )
        if not page.get_contents():
            continue
        # assumes that two column pages do not contain important information
        if is_two_column_page(page):
            logger.debug(f"Page excluded: {page.number=}")
            continue
        page = mask_sensitive_info(page)
        pages.append(page)
    return pages


def concat_pdfs(paths: list[str], output_path: str) -> tuple[list[int], list[int]]:
    docs = [fitz.open(path) for path in paths]
    pages: list[fitz.Page] = []
    page_counts = []
    for doc in docs:
        pdf_pages = extract_pdf_pages(doc)
        pages.extend(pdf_pages)
        page_counts.append(len(pdf_pages))

    # pad half an inch of margin to the left and right of the widest page
    max_width = max(page.rect.width for page in pages) + 72
    output_pdf = fitz.open()
    for page in pages:
        original_width = page.rect.width
        original_height = page.rect.height
        # pad to the widest page for more consistent processing
        new_page = output_pdf.new_page(width=max_width, height=original_height)
        x_offset = (max_width - original_width) / 2
        new_page_rect = fitz.Rect(
            x_offset, 0, x_offset + original_width, original_height
        )
        new_page.show_pdf_page(new_page_rect, page.parent, page.number)
    output_pdf.save(output_path, garbage=4, clean=True, deflate=True)
    output_pdf.close()

    for doc in docs:
        doc.close()

    page_ends = list(accumulate(page_counts))
    page_starts = [0] + list(page_ends[:-1])
    return page_starts, page_ends
