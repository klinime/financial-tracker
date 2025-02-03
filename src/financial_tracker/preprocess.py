import csv
import io
import json
import logging
import os

import dateutil  # type: ignore[import-untyped]
import pandas as pd


def is_string_form(header: str) -> bool:
    # allow empty strings
    if not header:
        return True

    # dates are not strings
    try:
        dateutil.parser.parse(header)
        return False
    except Exception:
        pass

    headers_to_try = [header]
    if not header[0].isdigit():
        # e.g. dollar sign
        headers_to_try.append(header[1:])
    for header in headers_to_try:
        # numeric values are not strings
        for dtype in [float, int]:
            try:
                dtype(header)
                return False
            except Exception:
                pass
            try:
                dtype(header.replace(",", ""))
                return False
            except Exception:
                pass
    return True


def split_df_columns(columns: list[str | int], target_num_cols: int) -> list[str | int]:
    # naive attempt to split columns that should've been split
    columns = [column for column in columns if column or column == 0]
    while len(columns) < target_num_cols:
        if not any(isinstance(column, str) and " " in column for column in columns):
            break
        for i, column in enumerate(columns):
            # expand columns one by one
            if isinstance(column, str) and " " in column:
                split_column = column.split(" ")
                columns[i : i + 1] = [split_column[0], " ".join(split_column[1:])]
    if len(columns) > target_num_cols:
        columns = columns[:target_num_cols]
    return columns


def squeeze_df(df: pd.DataFrame) -> pd.DataFrame:
    # squeeze NA values from df and return if the resulting df is of valid shape
    if df.empty:
        return df
    non_nan_values = [cell for cell in df.stack().tolist() if cell]
    if len(non_nan_values) == df.size:
        return df

    # squeezing NA values only reduces the number of columns and not rows
    num_rows = df.shape[0]
    if len(non_nan_values) % num_rows != 0:
        return pd.DataFrame()
    num_cols = len(non_nan_values) // num_rows
    # format back with original indices and columns
    return pd.DataFrame(
        [
            non_nan_values[i : i + num_cols]
            for i in range(0, len(non_nan_values), num_cols)
        ],
        index=df.index,
        columns=split_df_columns(df.columns, num_cols),
    )


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # drop empty columns
    df.replace("", None, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    columns = split_df_columns(df.columns, df.columns.size)
    if len(columns) == df.columns.size:
        df.columns = columns
    squeezed_df = squeeze_df(df)
    if not squeezed_df.empty:
        df = squeezed_df
    return df


def clean_string(string: str) -> str:
    return string.strip().replace("\xa0", " ")


class PDFTextBuilder:
    def __init__(self, data_dir: str):
        self.logger = logging.getLogger(__name__)
        self.extract_dir = os.path.join(data_dir, "extract")
        data = json.load(open(os.path.join(self.extract_dir, "structuredData.json")))
        self.elements = [
            element
            for element in data["elements"]
            if "filePaths" in element or "Text" in element
        ]
        self.metadata = list(
            json.load(open(os.path.join(data_dir, "metadata.json"))).values()
        )
        self.category = self.metadata[0]["category"]
        self.doc_num = 0
        self.pdf_text = f"<{self.category}>\n<doc{self.doc_num}>\n<page0>\n"

    def tag_metadata(self, page: int, next_page: int) -> None:
        # append metadata as html like tags to self.pdf_text
        # metadata is in the order of category -> document -> page
        if next_page == -1:
            self.pdf_text += f"</page{self.metadata[-1]['page_end'] - 1}>\n"
            self.pdf_text += f"</doc{len(self.metadata) - 1}>\n"
            self.pdf_text += f"</{self.metadata[-1]['category']}>\n"
            return
        tag_text = ""
        if page + 1 == next_page:
            tag_text += f"</page{page}>\n"
            if next_page == self.metadata[self.doc_num]["page_end"]:
                tag_text += f"</doc{self.doc_num}>\n"
                self.doc_num += 1
                next_category = self.metadata[self.doc_num]["category"]
                if self.category != next_category:
                    tag_text += f"</{self.category}>\n"
                    tag_text += f"<{next_category}>\n"
                    self.category = next_category
                tag_text += f"<doc{self.doc_num}>\n"
            tag_text += f"<page{next_page}>\n"
        self.pdf_text += tag_text

    def process_table(self, file_path: str) -> str:
        # append table as a string to self.pdf_text
        table_text = ""
        merged_header = []
        merged_text = ""
        path = os.path.join(self.extract_dir, file_path)
        read_csv_kwargs = {
            "header": None,
            "dtype": str,
            "keep_default_na": False,
            "na_filter": False,
            "index_col": False,
        }
        try:
            df = pd.read_csv(path, **read_csv_kwargs)
        except Exception:
            try:
                # first row could be a merged header which tends to be problematic when parsing
                df = pd.read_csv(path, **read_csv_kwargs, skiprows=1)
                with open(path, encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    merged_header = [
                        cell for cell in map(clean_string, next(reader)) if cell
                    ]
                merged_text = "\n" + " ".join(merged_header).strip()
                table_text = merged_text.strip()
            except Exception:
                self.pdf_text += f"Raw table: {file_path}\n"
                return table_text

        df = df.map(clean_string)
        df = clean_df(df)
        # extract plain text from table for matching with text blocks afterwards
        table_text += " ".join(df.stack().reset_index(drop=True)).strip()

        has_header = all(is_string_form(header) for header in df.iloc[0])
        has_index = all(is_string_form(index) for index in df.iloc[:, 0])

        # preprocess table header and index to prepare for json parsing
        if has_header:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df = clean_df(df)
            if has_index:
                df = df.set_index(df.columns[0])
                df = clean_df(df)
            # achieve unique column names by prepending merged headers
            if len(merged_header) > 1 and df.columns.size > len(merged_header):
                merge_size = df.columns.size // len(merged_header)
                df.columns = [
                    f"{merged_header[i // merge_size]} {column}"
                    for i, column in enumerate(df.columns)
                ]
                df = clean_df(df)

        # parse table to json
        orient = (
            "index"
            if has_header and has_index
            else "records" if has_header else "values"
        )
        try:
            self.pdf_text += f"Parsed table: {file_path}{merged_text}\n"
            self.pdf_text += json.dumps(json.loads(df.to_json(orient=orient)), indent=2)
        except Exception:
            # fallback to csv
            csv_string = io.StringIO()
            df.to_csv(csv_string)
            self.pdf_text += f"Semi-parsed table: {file_path}{merged_text}\n"
            self.pdf_text += csv_string.getvalue()
            csv_string.close()
        self.pdf_text += "\n"
        return table_text

    def process_text(
        self, text: str, next_text: str, table_text: str | None
    ) -> str | None:
        # append text to self.pdf_text if not part of a table
        if table_text:
            text = text.strip().replace("\xa0", " ")
            if table_text.startswith(text):
                table_text = table_text[len(text) :].strip()
                return table_text
            elif next_text:
                # for some reason, pdf processing sometimes swaps the order of texts
                # in this case, we remove the next text and keep the current text
                next_text = next_text.strip().replace("\xa0", " ")
                if table_text.startswith(next_text):
                    next_table_text = table_text[len(next_text) :].strip()
                    if next_table_text.startswith(text):
                        return (
                            table_text[: len(next_text)] + next_table_text[len(text) :]
                        )
        else:
            self.pdf_text += text + "\n"
            return table_text
        self.logger.error(
            f"Error processing text: {text=}, {next_text=}, {table_text=}"
        )
        return None

    def process_pdf_data(self) -> str:
        table_text: str | None = None
        for i, element in enumerate(self.elements):
            next_element = self.elements[i + 1] if i < len(self.elements) - 1 else None
            if "filePaths" in element:
                table_text = self.process_table(element["filePaths"][0])
            elif "Text" in element:
                next_text = (
                    next_element["Text"]
                    if next_element and "Text" in next_element
                    else ""
                )
                table_text = self.process_text(element["Text"], next_text, table_text)
                if table_text is None:
                    self.logger.error(f"Error processing text: {i=}, {element=}")
                    return ""
            self.tag_metadata(
                element["Page"], next_element["Page"] if next_element else -1
            )
        return self.pdf_text
