import logging
import re
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


def page_blocks_equivalent(
    page_blocks1: list[dict[str, Any]], page_blocks2: list[dict[str, Any]]
) -> bool:
    return all(
        text_block_equivalent(block1, block2)
        for block1, block2 in zip(page_blocks1, page_blocks2)
    )


def sorted_header_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # sort by bottom of the block because excluding header goes from bottom to top
    return sorted(text_blocks, key=lambda block: block["bbox"][3])


def sorted_footer_blocks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # sort by top of the block because excluding footer goes from top to bottom
    return sorted(text_blocks, key=lambda block: block["bbox"][1], reverse=True)


def pdf_margin_offsets(doc: fitz.Document) -> tuple[int, int]:
    logger = logging.getLogger(__name__)

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
    header_offsets = []
    footer_offsets = []
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
        header_offset = -1
        footer_offset = -1
        # equivalent to prefix / suffix matches against each reference text block
        for ref_text_blocks in cur_ref_blocks:
            for ref_text_block, page_text_block in zip(
                sorted_header_blocks(ref_text_blocks),
                sorted_header_blocks(page_text_blocks),
            ):
                if text_block_equivalent(ref_text_block, page_text_block):
                    header_offset = page_text_block["bbox"][3]
                else:
                    break
            for ref_text_block, page_text_block in zip(
                sorted_footer_blocks(ref_text_blocks),
                sorted_footer_blocks(page_text_blocks),
            ):
                if text_block_equivalent(ref_text_block, page_text_block):
                    footer_offset = page_text_block["bbox"][1]
                else:
                    break
        if header_offset != -1:
            header_offsets.append(header_offset)
        if footer_offset != -1:
            footer_offsets.append(footer_offset)

    header_offset = np.median(header_offsets) if header_offsets else 0
    footer_offset = np.median(footer_offsets) if footer_offsets else doc[0].rect.height
    logger.debug(f"{header_offset=} {footer_offset=}")
    for page_num in range(ref_page_count):
        logger.debug(
            f"removed_header:\n{doc[page_num].get_text(clip=fitz.Rect(0, 0, doc[page_num].rect.width, header_offset))}"
        )
        logger.debug(
            f"removed_footer:\n{doc[page_num].get_text(clip=fitz.Rect(0, footer_offset, doc[page_num].rect.width, doc[page_num].rect.height))}"
        )

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


def extract_pdf_pages(doc: fitz.Document) -> list[fitz.Page]:
    logger = logging.getLogger(__name__)

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
        pages.append(page)
    return pages


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    import pathlib

    docs = [
        fitz.open(str(pdf_path))
        for pdf_path in pathlib.Path("../../data").glob("*.pdf", case_sensitive=False)
    ]
    pages: list[fitz.Page] = []
    for doc in docs:
        pages.extend(extract_pdf_pages(doc))

    output_pdf = fitz.open()
    for page in pages:
        new_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(new_page.rect, page.parent, page.number)
    output_pdf.save("output.pdf")
    output_pdf.close()

    for doc in docs:
        doc.close()
