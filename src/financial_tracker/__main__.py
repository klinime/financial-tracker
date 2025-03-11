import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

from financial_tracker.categories import statement_categories

load_dotenv()


def concat_statements(data_dir: Path, **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    categories = statement_categories()
    statements_paths = [
        str(data_dir / f"{category}_statement.pdf") for category in categories
    ]
    metadata_paths = [
        str(data_dir / f"{category}_metadata.json") for category in categories
    ]

    from financial_tracker.pdf import concat_pdfs

    dirs = [data_dir / category for category in categories]
    globs = [list(dir.glob("*.pdf", case_sensitive=False)) for dir in dirs]
    for glob, statements_path, metadata_path in zip(
        globs, statements_paths, metadata_paths
    ):
        if os.path.exists(statements_path) and os.path.exists(metadata_path):
            logger.info(
                f"Statements and metadata already exist in {statements_path=} and {metadata_path=}"
            )
            continue
        page_starts, page_ends = concat_pdfs(
            [str(path) for path in glob], statements_path
        )
        logger.info(f"Concatenated statements saved to {statements_path=}")
        metadata = {
            path: {
                "page_start": start,
                "page_end": end,
            }
            for path, start, end in zip(statements_paths, page_starts, page_ends)
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path=}")


def extract_data(data_dir: Path, **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    logging.getLogger("adobe.pdfservices").setLevel(logging.ERROR)
    categories = statement_categories()
    statements_paths = [
        str(data_dir / f"{category}_statement.pdf") for category in categories
    ]
    extract_dirs = [str(data_dir / f"{category}_extract") for category in categories]

    from financial_tracker.pdf_services import PDFTextTableExtractor

    extractor = PDFTextTableExtractor()
    for statements_path, extract_dir in zip(statements_paths, extract_dirs):
        if os.path.exists(extract_dir):
            logger.info(f"Extracted data already exists in {extract_dir=}")
            continue
        extractor.extract_text_table(statements_path, extract_dir)
        logger.info(f"Extracted data saved to {extract_dir=}")


def process_statements(data_dir: Path, **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    categories = statement_categories()
    extract_dirs = [str(data_dir / f"{category}_extract") for category in categories]
    metadata_paths = [
        str(data_dir / f"{category}_metadata.json") for category in categories
    ]
    pdf_text_paths = [
        str(data_dir / f"{category}_pdf_text.txt") for category in categories
    ]

    from financial_tracker.preprocess import PDFTextBuilder

    for extract_dir, metadata_path, pdf_text_path in zip(
        extract_dirs, metadata_paths, pdf_text_paths
    ):
        if os.path.exists(pdf_text_path):
            logger.info(f"PDF text already exists in {pdf_text_path=}")
            continue
        pdf_text_builder = PDFTextBuilder(extract_dir, metadata_path)
        pdf_text = pdf_text_builder.process_pdf_data()
        with open(pdf_text_path, "w") as f:
            f.write(pdf_text)
        logger.info(f"PDF text saved to {pdf_text_path=}")


def analyze_statements(data_dir: Path, examples_path: Path, **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    categories = statement_categories()
    pdf_text_paths = [
        str(data_dir / f"{category}_pdf_text.txt") for category in categories
    ]
    transactions_paths = [
        str(data_dir / f"{category}_transactions.json") for category in categories
    ]

    from financial_tracker.llm import FinancialLLM

    for pdf_text_path, transactions_path in zip(pdf_text_paths, transactions_paths):
        if os.path.exists(transactions_path):
            logger.info(f"Transactions already exists in {transactions_path=}")
            continue
        llm = FinancialLLM(str(examples_path))
        response = llm.generate_text(pdf_text_path)
        with open(transactions_path, "w") as f:
            json.dump(json.loads(response), f, indent=4)
        logger.info(f"Transactions saved to {transactions_path=}")


def correct_transactions(data_dir: Path, **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    categories = statement_categories()
    transactions_paths = [
        str(data_dir / f"{category}_transactions.json") for category in categories
    ]

    from financial_tracker.correction import TransactionCorrector

    for transactions_path, category in zip(transactions_paths, categories):
        corrected_path = transactions_path.replace(".json", "_corrected.json")
        if os.path.exists(corrected_path):
            logger.info(f"Transactions already corrected in {transactions_path=}")
            continue
        transaction_corrector = TransactionCorrector(transactions_path, category)
        transactions = transaction_corrector.correct_transactions()
        with open(corrected_path, "w") as f:
            json.dump(transactions, f, indent=4)
        logger.info(f"Transactions saved to {corrected_path=}")


def visualize_statements(data_dirs: list[Path], **kwargs: Any) -> None:
    logger = logging.getLogger(__name__)
    categories = statement_categories()
    transactions_paths = [
        [
            str(data_dir / f"{category}_transactions_corrected.json")
            for category in categories
        ]
        for data_dir in data_dirs
    ]

    if not all(
        os.path.exists(path)
        for transactions_path in transactions_paths
        for path in transactions_path
    ):
        logger.info(f"Transactions not found in {transactions_paths=}")
        return

    from financial_tracker.visualize import TransactionVisualizer

    transactions = [
        [json.load(open(path)) for path in transactions_path]
        for transactions_path in transactions_paths
    ]
    transaction_visualizer = TransactionVisualizer(transactions)
    transaction_visualizer.build_app()
    transaction_visualizer.run()


class KwArgs(TypedDict, total=False):
    """General kwargs for specifying argparse arguments."""

    type: type
    nargs: str | int
    default: Any
    required: bool
    help: str


def stage_info() -> dict[str, dict[str, Any]]:
    return {
        "concat": {
            "help": "Concatenate statements",
            "action": concat_statements,
        },
        "extract": {
            "help": "Extract data from statements",
            "action": extract_data,
        },
        "preprocess": {
            "help": "Preprocess statements",
            "action": process_statements,
        },
        "analyze": {
            "help": "Analyze statements",
            "action": analyze_statements,
        },
        "correct": {
            "help": "Correct transactions",
            "action": correct_transactions,
        },
    }


def parse_args() -> argparse.Namespace:
    data_dir_args = ["--data-dir"]
    data_dir_kwargs: KwArgs = {
        "type": str,
        "default": "data",
        "help": "Path to the data directory",
    }
    data_dirs_args = ["--data-dirs"]
    data_dirs_kwargs: KwArgs = {
        "type": str,
        "nargs": "+",
        "help": "Path(s) to the data directory",
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
    for stage, info in stage_info().items():
        parser_stage = parser_sub.add_parser(stage, help=info["help"])
        parser_stage.add_argument(*data_dir_args, **data_dir_kwargs)
        if stage == "analyze":
            parser_stage.add_argument(*examples_args, **examples_kwargs)

    parser_multistage = parser_sub.add_parser("multistage", help="Run multiple stages")
    parser_multistage.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=list(stage_info().keys()),
        help="Stages to run",
    )
    parser_multistage.add_argument(*data_dirs_args, **data_dirs_kwargs)
    parser_multistage.add_argument(*examples_args, **examples_kwargs)

    parser_visualize = parser_sub.add_parser("visualize", help="Visualize statements")
    parser_visualize.add_argument(*data_dirs_args, **data_dirs_kwargs)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command

    if command == "visualize":
        visualize_statements([Path(data_dir) for data_dir in args.data_dirs])
    elif command == "multistage":
        for stage in args.stages:
            for data_dir in args.data_dirs:
                stage_info()[stage]["action"](
                    Path(data_dir), args.examples_path, args.data_dir
                )
    else:
        data_dir = Path(args.data_dir)
        stage_info()[command]["action"](data_dir, args.examples_path)


if __name__ == "__main__":
    main()
