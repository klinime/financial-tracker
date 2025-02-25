import csv
import json
import os
from io import StringIO
from itertools import takewhile
from logging import getLogger
from math import ceil

import dateutil  # type: ignore[import-untyped]
import pandas as pd


def is_date_form(header: str) -> bool:
    try:
        dateutil.parser.parse(header)
        return True
    except Exception:
        return False


def is_string_form(header: str) -> bool:
    # allow empty strings
    if not header:
        return True

    # dates are not strings
    if is_date_form(header):
        return False

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


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # drop empty rows and columns
    df.replace("", None, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    columns = split_df_columns(df.columns, df.columns.size)
    if len(columns) == df.columns.size:
        df.columns = columns
    return df


def clean_string(string: str) -> str:
    return string.strip().replace("\xa0", " ")


class PDFTextBuilder:
    def __init__(self, data_dir: str):
        self.logger = getLogger(__name__)
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
        # append metadata as xml like tags to self.pdf_text
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

    def read_csv(self, file_path: str, merged_header: list[str]) -> pd.DataFrame:
        file_path = os.path.join(self.extract_dir, file_path)
        read_csv_kwargs = {
            "header": None,
            "dtype": str,
            "keep_default_na": False,
            "na_filter": False,
            "index_col": False,
        }
        try:
            df = pd.read_csv(file_path, **read_csv_kwargs)
        except Exception:
            # first couple of rows could be a merged header which tends to be problematic when parsing
            # or the headers are accidentally split into multiple rows
            with open(file_path, encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                rows = [list(map(clean_string, row)) for row in reader]
            col_counts = [len(row) for row in rows]
            expected_col_count = max(col_counts, key=col_counts.count)
            # leading rows that do not have the expected number of columns
            merged_header_rows = sum(
                1 for _ in takewhile(lambda x: x != expected_col_count, col_counts)
            )
            for row in rows[:merged_header_rows]:
                for cell in row:
                    if cell:
                        merged_header.append(cell)
            # leading rows that when combined, have the expected number of columns
            split_header_rows = 0
            split_headers = []
            for row in rows:
                split_header_rows += 1
                split_headers.extend([cell for cell in row if cell])
                if len(split_headers) == expected_col_count:
                    break
                elif len(split_headers) > expected_col_count:
                    split_header_rows = 0
                    split_headers = []
                    break

            if merged_header_rows:
                # scenario 1: the headers are merged
                df = pd.read_csv(
                    file_path, **read_csv_kwargs, skiprows=merged_header_rows
                )
                if split_header_rows:
                    # scenario 2: the headers are split into multiple rows
                    if split_header_rows > merged_header_rows:
                        offset = split_header_rows - merged_header_rows
                        df.iloc[offset - 1] = split_headers
                        df = df.iloc[offset - 1 :]
                        del merged_header[:]
                    else:
                        df = df.iloc[merged_header_rows:]
                        df = pd.concat([pd.DataFrame(split_headers), df])
                    df.reset_index(drop=True, inplace=True)
        return df

    def merge_split_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # some rows are accidentally split into multiple rows with disjoint columns
        df_rows: list[pd.Series] = []
        concat_row = df.iloc[0].copy()
        row_not_na = df.iloc[0].notna()
        for i in range(1, df.shape[0]):
            # merge adjacent rows where non-None entries are disjoint
            next_not_na = df.iloc[i].notna()
            if (row_not_na & next_not_na).any():
                df_rows.append(concat_row)
                concat_row = df.iloc[i].copy()
                row_not_na = df.iloc[i].notna()
            else:
                concat_row[next_not_na] = df.iloc[i][next_not_na]
                row_not_na |= next_not_na
        df_rows.append(concat_row)
        if df.shape[0] != len(df_rows):
            df = pd.DataFrame(df_rows)
            df.reset_index(drop=True, inplace=True)
        return df

    def merge_split_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # some columns are accidentally split into multiple columns at some offset
        # pop until one of the rows has no None entries to reduce the number of columns
        df_rows = []
        na_to_pop = df.isna().sum(axis=1).min()
        for row in df.itertuples(index=False):
            row_list = list(row)
            for _ in range(na_to_pop):
                row_list.remove(None)
            df_rows.append(row_list)
        df = pd.DataFrame(df_rows)
        # for some reason, there are some columns that are all NaN except for one row
        # try to pull them into the previous column until it is part of a normal column
        not_na_per_col = df.iloc[1:].notna().sum(axis=0)
        for i in reversed(range(df.shape[1] - 1)):
            if not_na_per_col.iloc[i] == 1 and df.iloc[0, i + 1] is None:
                df.iloc[0, i + 1] = df.iloc[0, i]
                df = df.drop(df.columns[i], axis=1)
        return df

    def process_table(self, file_path: str) -> str:
        # append table as a string to self.pdf_text
        merged_header: list[str] = []
        try:
            df = self.read_csv(file_path, merged_header)
        except Exception:
            self.pdf_text += f"Raw table: {file_path}\n"
            return ""
        merged_text = ""
        table_text = ""
        if merged_header:
            merged_text = "\n" + " ".join(merged_header).strip()
            table_text = merged_text.strip()

        df = df.map(clean_string)
        df = clean_df(df)
        # extract plain text from table for matching with text blocks afterwards
        table_text += " ".join(df.stack().reset_index(drop=True)).strip()

        df.columns = range(df.columns.size)
        df = self.merge_split_rows(df)
        df = self.merge_split_columns(df)

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
            if merged_header:
                if len(merged_header) == df.columns.isna().sum():
                    # fill in missing columns with merged headers
                    columns = df.columns.to_numpy()
                    columns[df.columns.isna()] = merged_header
                    df.columns = columns
                elif (
                    len(merged_header) > 1
                    and not df.columns.is_unique
                    and df.columns.size > len(merged_header)
                ):
                    # achieve unique column names by prepending merged headers
                    merge_size = ceil(df.columns.size / len(merged_header))
                    df.columns = [
                        f"{merged_header[i // merge_size]} {column}"
                        for i, column in enumerate(df.columns)
                    ]

        # empty after processing
        if df.empty:
            self.pdf_text += f"Raw table: {file_path}\n"
            return ""
        # if the table is too sparse, it's probably not a well-formed table
        if df.isna().values.sum() / df.size > 0.2 and df.isna().sum(axis=0).min() > 0:
            self.pdf_text += f"Raw table: {file_path}\n"
            return ""

        self.pdf_text += f"Parsed table: {file_path}{merged_text}\n"
        if df.shape[0] > 0:
            first_row = df.iloc[0]
            if (
                len(first_row) > 6
                and is_date_form(first_row.iloc[0])
                and first_row.iloc[1:6].isna().all()
            ):
                # assume is calendar, which we don't need to parse
                return table_text
        try:
            # parse table to json
            orient = (
                "index"
                if has_header and has_index
                else "records" if has_header else "values"
            )
            self.pdf_text += df.to_json(orient=orient, indent=2)
        except Exception:
            # fallback to csv
            csv_string = StringIO()
            df.to_csv(csv_string)
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
        table_text: str | None = ""
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
