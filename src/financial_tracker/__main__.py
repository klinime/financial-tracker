import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from financial_tracker.pdf import concat_pdfs
from financial_tracker.pdf_services import PDFTextTableExtractor
from financial_tracker.preprocess import PDFTextBuilder

load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("adobe.pdfservices").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    data_dir = Path("data")
    categories = ["income", "bank", "expense"]
    dirs = [data_dir / category for category in categories]
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
            "category": categories[cat_index],
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
        extractor = PDFTextTableExtractor(
            str(data_dir / "pdfservices-api-credentials.json")
        )
        extractor.extract_text_table(statements_path, extract_dir)
        logger.info(f"Extracted data saved to {extract_dir=}")
    pdf_text_builder = PDFTextBuilder(str(data_dir))
    pdf_text = pdf_text_builder.process_pdf_data()
    pdf_text_path = str(data_dir / "pdf_text.txt")
    with open(pdf_text_path, "w") as f:
        f.write(pdf_text)
    logger.info(f"PDF text saved to {pdf_text_path=}")


if __name__ == "__main__":
    main()
