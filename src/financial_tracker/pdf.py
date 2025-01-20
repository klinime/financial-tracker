import numpy as np
import pdfplumber

point_threshold = 4.5  # 4.5 points is 1/16 inches


def is_same_line(obj1: pdfplumber.page.T_obj, obj2: pdfplumber.page.T_obj) -> bool:
    assert isinstance(obj1["top"], float)
    assert isinstance(obj2["top"], float)
    assert isinstance(obj1["bottom"], float)
    assert isinstance(obj2["bottom"], float)
    return (abs(obj1["top"] - obj2["top"]) < point_threshold) and (
        abs(obj1["bottom"] - obj2["bottom"]) < point_threshold
    )


def is_h_aligned_center(
    obj1: pdfplumber.page.T_obj, obj2: pdfplumber.page.T_obj
) -> bool:
    assert isinstance(obj1["top"], float)
    assert isinstance(obj2["top"], float)
    assert isinstance(obj1["bottom"], float)
    assert isinstance(obj2["bottom"], float)
    return (
        abs((obj1["top"] + obj1["bottom"]) - (obj2["top"] + obj2["bottom"]))
        < point_threshold * 2
    )


def is_aligned_center(obj1: pdfplumber.page.T_obj, obj2: pdfplumber.page.T_obj) -> bool:
    assert isinstance(obj1["x0"], float)
    assert isinstance(obj2["x0"], float)
    assert isinstance(obj1["x1"], float)
    assert isinstance(obj2["x1"], float)
    return (
        abs((obj1["x0"] + obj1["x1"]) - (obj2["x0"] + obj2["x1"])) < point_threshold * 2
    )


def is_aligned_left(obj1: pdfplumber.page.T_obj, obj2: pdfplumber.page.T_obj) -> bool:
    assert isinstance(obj1["x0"], float)
    assert isinstance(obj2["x0"], float)
    return abs(obj1["x0"] - obj2["x0"]) < point_threshold


def is_equivalent_words(
    word1: pdfplumber.page.T_obj, word2: pdfplumber.page.T_obj
) -> bool:
    # relaxes the requirement that the words are exactly the same
    # to allow for numeric values matches for patterns like "page 1 of 3"
    return is_aligned_center(word1, word2) and (
        (word1["text"] == word2["text"])
        or (
            word1["text"].isnumeric()
            and word2["text"].isnumeric()
            and (len(word1["text"]) == len(word2["text"]))
        )
    )


def categorize_two_columns(
    words: list[pdfplumber.page.T_obj],
) -> tuple[list[pdfplumber.page.T_obj], list[pdfplumber.page.T_obj]]:
    min_col_gap = 144  # 144 points is 2 inches
    left_col: list[list[pdfplumber.page.T_obj]] = []
    right_col: list[list[pdfplumber.page.T_obj]] = []
    left_col_words: list[pdfplumber.page.T_obj] = [words[0]]
    right_col_words: list[pdfplumber.page.T_obj] = []
    # cartesian product of scenarios:
    # [left column, right column] x [start x [prev end left, prev end right], continuation]
    for word in words[1:]:
        if (
            left_col_words
            and is_same_line(left_col_words[-1], word)
            and abs(word["x0"] - left_col_words[-1]["x1"]) < min_col_gap
        ):
            # left column x continuation
            left_col_words.append(word)
        elif (
            not right_col_words
            and is_same_line(left_col_words[-1], word)
            and (not right_col or is_aligned_left(right_col[0][0], word))
        ):
            # right column x start x prev end left
            right_col_words.append(word)
        elif (
            right_col_words
            and is_same_line(right_col_words[-1], word)
            and abs(word["x0"] - right_col_words[-1]["x1"]) < min_col_gap
        ):
            # right column x continuation
            right_col_words.append(word)
        elif right_col_words and is_aligned_left(left_col_words[0], word):
            # left column x start x prev end right
            left_col.append(left_col_words)
            right_col.append(right_col_words)
            left_col_words = [word]
            right_col_words = []
        elif not right_col_words and is_aligned_left(left_col_words[0], word):
            # left column x start x prev end left
            left_col.append(left_col_words)
            left_col_words = [word]
        elif (
            right_col_words
            and not is_same_line(right_col_words[-1], word)
            and is_aligned_left(right_col[0][0], word)
        ):
            # right column x start x prev end right
            right_col.append(right_col_words)
            right_col_words = [word]
        else:
            break

    # append the remaining lines
    if left_col_words:
        left_col.append(left_col_words)
    if right_col_words:
        right_col.append(right_col_words)

    print("\nDebug:")
    print("Left column:")
    for left_line in left_col:
        print(" ".join(word["text"] for word in left_line))
    print("Right column:")
    for right_line in right_col:
        print(" ".join(word["text"] for word in right_line))

    # hardcoded assumption that there are at least two lines in each column
    if len(left_col) < 2:
        return [], []
    return [word for col_words in left_col for word in col_words], [
        word for col_words in right_col for word in col_words
    ]


def get_fixed_offset(
    objs1: list[pdfplumber.page.T_obj],
    objs2: list[pdfplumber.page.T_obj],
    header: bool = True,
) -> float:
    candidate_objects = []
    for obj1, obj2 in zip(
        objs1 if header else objs1[::-1], objs2 if header else objs2[::-1]
    ):
        if is_equivalent_words(obj1, obj2):
            candidate_objects.append(obj1)
            candidate_objects.append(obj2)
        else:
            # to be considered a candidate for header/footer, entire lines must be equivalent
            while candidate_objects:
                if is_same_line(candidate_objects[-1], obj1) or is_same_line(
                    candidate_objects[-1], obj2
                ):
                    candidate_objects.pop()
                else:
                    break
            break
    if not candidate_objects:
        return 0 if header else -1
    bbox = pdfplumber.utils.objects_to_bbox(candidate_objects)
    assert isinstance(bbox[1], float)
    assert isinstance(bbox[3], float)
    return bbox[3] if header else bbox[1]  # bottom or top


def pdf_header_footer_offsets(pages: list[pdfplumber.page.Page]) -> tuple[int, int]:
    page0_words = pages[0].extract_words()
    page0_header_left_col, page0_header_right_col = categorize_two_columns(page0_words)
    candidate_header_offsets = []
    candidate_footer_offsets = []
    for page in pages[1:]:
        words = page.extract_words()
        header_left_col, header_right_col = categorize_two_columns(words)
        left_header_offset = get_fixed_offset(page0_header_left_col, header_left_col)
        if page0_header_right_col or header_right_col:
            right_header_offset = get_fixed_offset(
                (
                    page0_header_right_col
                    if page0_header_right_col
                    else page0_header_left_col
                ),
                header_right_col if header_right_col else header_left_col,
            )
        else:
            right_header_offset = 0
        candidate_header_offsets.append(max(left_header_offset, right_header_offset))

        # assumes that the footer has a single column or can be effectively treated as a single column
        candidate_footer_offsets.append(
            get_fixed_offset(page0_words, words, header=False)
        )

    return np.median(candidate_header_offsets), np.median(candidate_footer_offsets)


if __name__ == "__main__":
    import glob

    for pdf_path in glob.glob("../../data/*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            print("\nProcessing:", pdf_path)
            header_offset, footer_offset = pdf_header_footer_offsets(pdf.pages)
            print("\nHeader offset:", header_offset, "Footer offset:", footer_offset)
