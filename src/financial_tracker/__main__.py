import json
import logging
import pathlib

from financial_tracker.pdf import concat_pdfs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_dir = pathlib.Path("../../data")
    categories = ["income", "brokerage", "bank", "expense"]
    dirs = [data_dir / category for category in categories]
    globs = [list(dir.glob("*.pdf", case_sensitive=False)) for dir in dirs]
    cat_indices = sum([[i] * len(glob) for i, glob in enumerate(globs)], [])
    paths = [str(path) for glob in globs for path in glob]
    statements_path = data_dir / "statements.pdf"
    page_starts, page_ends = concat_pdfs(paths, str(statements_path))
    logger.info(f"Concatenated statements saved to {statements_path}")
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                path: {
                    "page_start": page_start,
                    "page_end": page_end,
                    "category": categories[cat_index],
                }
                for path, page_start, page_end, cat_index in zip(
                    paths, page_starts, page_ends, cat_indices
                )
            },
            f,
            indent=4,
        )
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
