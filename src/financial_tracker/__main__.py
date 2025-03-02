import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from financial_tracker.llm import FinancialLLM
from financial_tracker.pdf import concat_pdfs
from financial_tracker.pdf_services import PDFTextTableExtractor
from financial_tracker.preprocess import PDFTextBuilder

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["income", "bank", "expense"],
        help="List of categories of the statements",
    )
    parser.add_argument(
        "--examples-path",
        type=str,
        default="data/examples.txt",
        help="Path to the examples file",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("adobe.pdfservices").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    args = parse_args()
    data_dir = Path(args.data_dir)
    dirs = [data_dir / category for category in args.categories]
    globs = [list(dir.glob("*.pdf", case_sensitive=False)) for dir in dirs]
    cat_indices = sum([[i] * len(glob) for i, glob in enumerate(globs)], [])
    paths = [str(path) for glob in globs for path in glob]
    statements_path = str(data_dir / "statements.pdf")
    page_starts, page_ends = concat_pdfs(paths, statements_path)
    logger.info(f"Concatenated statements saved to {statements_path=}")
    metadata_path = str(data_dir / "metadata.json")
    metadata = {
        path: {
            "page_start": page_start,
            "page_end": page_end,
            "category": args.categories[cat_index],
        }
        for path, page_start, page_end, cat_index in zip(
            paths, page_starts, page_ends, cat_indices
        )
    }

    extract_dir = str(data_dir / "extract")
    if os.path.exists(metadata_path) and metadata == json.load(open(metadata_path)):
        logger.info(f"Metadata already exists in {metadata_path=}")
        logger.info(f"Loading extracted data from {extract_dir=}")
    else:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path=}")
        logger.info("Extracting data with Adobe PDF Services...")
        extractor = PDFTextTableExtractor()
        extractor.extract_text_table(statements_path, extract_dir)
        logger.info(f"Extracted data saved to {extract_dir=}")

    pdf_text_builder = PDFTextBuilder(str(data_dir))
    pdf_text = pdf_text_builder.process_pdf_data()
    pdf_text_path = str(data_dir / "pdf_text.txt")
    with open(pdf_text_path, "w") as f:
        f.write(pdf_text)
    logger.info(f"PDF text saved to {pdf_text_path=}")

    transactions_path = str(data_dir / "transactions.json")
    if os.path.exists(transactions_path):
        logger.info(f"Transactions already exists in {transactions_path=}")
    else:
        llm = FinancialLLM(str(data_dir / "examples.txt"))
        response = llm.generate_text(pdf_text)
        with open(transactions_path, "w") as f:
            json.dump(json.loads(response), f, indent=4)
        logger.info(f"Transactions saved to {transactions_path=}")


if __name__ == "__main__":
    main()
