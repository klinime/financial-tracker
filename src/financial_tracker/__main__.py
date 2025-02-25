import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

load_dotenv()


class KwArgs(TypedDict, total=False):
    """General kwargs for specifying argparse arguments."""

    type: type
    nargs: str | int
    default: Any
    required: bool
    help: str


def parse_args() -> argparse.Namespace:
    data_dir_args = ["--data-dir"]
    data_dir_kwargs: KwArgs = {
        "type": str,
        "default": "data",
        "help": "Path to the data directory",
    }
    categories_args = ["--categories"]
    categories_kwargs: KwArgs = {
        "type": str,
        "nargs": "+",
        "default": ["income", "bank", "expense"],
        "help": "List of categories of the statements",
    }
    examples_args = ["--examples-path"]
    examples_kwargs: KwArgs = {
        "type": str,
        "default": "data/examples.txt",
        "help": "Path to the examples file",
    }

    parser = argparse.ArgumentParser()
    parser_sub = parser.add_subparsers(
        title="financial_tracker subcommands", dest="command"
    )

    parser_concat = parser_sub.add_parser("concat", help="Concatenate statements")
    parser_concat.add_argument(*data_dir_args, **data_dir_kwargs)
    parser_concat.add_argument(*categories_args, **categories_kwargs)

    parser_extract = parser_sub.add_parser(
        "extract", help="Extract data from statements"
    )
    parser_extract.add_argument(*data_dir_args, **data_dir_kwargs)

    parser_preprocess = parser_sub.add_parser(
        "preprocess", help="Preprocess statements"
    )
    parser_preprocess.add_argument(*data_dir_args, **data_dir_kwargs)

    parser_analyze = parser_sub.add_parser("analyze", help="Analyze statements")
    parser_analyze.add_argument(*data_dir_args, **data_dir_kwargs)
    parser_analyze.add_argument(*examples_args, **examples_kwargs)

    parser_e2e = parser_sub.add_parser("e2e", help="Run all stages")
    parser_e2e.add_argument(*data_dir_args, **data_dir_kwargs)
    parser_e2e.add_argument(*categories_args, **categories_kwargs)
    parser_e2e.add_argument(*examples_args, **examples_kwargs)

    return parser.parse_args()


def concat_statements(data_dir: Path, categories: list[str]) -> None:
    logger = logging.getLogger(__name__)
    statements_path = str(data_dir / "statements.pdf")
    metadata_path = str(data_dir / "metadata.json")

    if os.path.exists(statements_path) and os.path.exists(metadata_path):
        logger.info(
            f"Statements and metadata already exist in {statements_path=} and {metadata_path=}"
        )
        return

    from financial_tracker.pdf import concat_pdfs

    dirs = [data_dir / category for category in categories]
    globs = [list(dir.glob("*.pdf", case_sensitive=False)) for dir in dirs]
    cat_indices = sum([[i] * len(glob) for i, glob in enumerate(globs)], [])
    paths = [str(path) for glob in globs for path in glob]
    page_starts, page_ends = concat_pdfs(paths, statements_path)
    logger.info(f"Concatenated statements saved to {statements_path=}")

    metadata = {
        path: {
            "page_start": page_start,
            "page_end": page_end,
            "category": categories[cat_index],
        }
        for path, page_start, page_end, cat_index in zip(
            paths, page_starts, page_ends, cat_indices
        )
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {metadata_path=}")


def extract_data(data_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logging.getLogger("adobe.pdfservices").setLevel(logging.ERROR)
    statements_path = str(data_dir / "statements.pdf")
    extract_dir = str(data_dir / "extract")

    if os.path.exists(extract_dir):
        logger.info(f"Extracted data already exists in {extract_dir=}")
        return

    from financial_tracker.pdf_services import PDFTextTableExtractor

    logger.info("Extracting data with Adobe PDF Services...")
    extractor = PDFTextTableExtractor()
    extractor.extract_text_table(statements_path, extract_dir)
    logger.info(f"Extracted data saved to {extract_dir=}")


def process_statements(data_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    pdf_text_path = str(data_dir / "pdf_text.txt")

    if os.path.exists(pdf_text_path):
        logger.info(f"PDF text already exists in {pdf_text_path=}")
        return

    from financial_tracker.preprocess import PDFTextBuilder

    pdf_text_builder = PDFTextBuilder(str(data_dir))
    pdf_text = pdf_text_builder.process_pdf_data()
    with open(pdf_text_path, "w") as f:
        f.write(pdf_text)
    logger.info(f"PDF text saved to {pdf_text_path=}")


def analyze_statements(data_dir: Path, examples_path: Path) -> None:
    logger = logging.getLogger(__name__)
    pdf_text_path = str(data_dir / "pdf_text.txt")
    transactions_path = str(data_dir / "transactions.json")

    if os.path.exists(transactions_path):
        logger.info(f"Transactions already exists in {transactions_path=}")
        return

    from financial_tracker.llm import FinancialLLM

    with open(pdf_text_path) as f:
        pdf_text = f.read()
    llm = FinancialLLM(str(examples_path))
    response = llm.generate_text(pdf_text)
    with open(transactions_path, "w") as f:
        json.dump(json.loads(response), f, indent=4)
    logger.info(f"Transactions saved to {transactions_path=}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    command = args.command
    data_dir = Path(args.data_dir)

    if command == "concat":
        concat_statements(data_dir, args.categories)
    elif command == "extract":
        extract_data(data_dir)
    elif command == "preprocess":
        process_statements(data_dir)
    elif command == "analyze":
        analyze_statements(data_dir, args.examples_path)
    elif command == "e2e":
        concat_statements(data_dir, args.categories)
        extract_data(data_dir)
        process_statements(data_dir)
        analyze_statements(data_dir, args.examples_path)


if __name__ == "__main__":
    main()
