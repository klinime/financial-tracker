
# financial-tracker

Financial tracking tool for bookkeeping, planning, projection, etc.
powered by large language models.

## Installation

```console
pip install build
python -m build
pip install dist/*.whl
pip install -e .
```

## Usage

Create a `.env` file in the root directory with the following variables:

```console
OPENAI_API_KEY=...
PDF_SERVICES_CLIENT_ID=...
PDF_SERVICES_CLIENT_SECRET=...
```

Download the relevant statements and categorize them into income, bank, expense
categories to place in subdirectories, as shown below:

```console
data/
  income/
    statement1.pdf
    statement2.pdf
  bank/
    statement1.pdf
    statement2.pdf
  expense/
    statement1.pdf
    statement2.pdf
```

To help the LLM understand some examples of the transactions, create a file
with examples of the transactions, explanations of how to classify, and the
expected output. An example of the file is shown below:

```console
data/examples.txt

Example 1:
[
  {
    "Date of Transaction": "12/01",
    "Transaction Description": "transaction 1",
    "Amount": "1.00"
  },
  {
    "Date of Transaction": "12/02",
    "Transaction Description": "transaction 2",
    "Amount": "2.00"
  },
  {
    "Date of Transaction": "12/03",
    "Transaction Description": "transaction 3",
    "Amount": "3.00"
  },
]

Reasoning:
The first transaction is a ..., so we categorize it as "...".
The second transaction is a ..., so we categorize it as "...".
The third transaction is a ..., so we categorize it as "...".

The extracted transactions are:
{
  "date": "2024-12-01",
  "amount": 1.00,
  "description": "transaction 1",
  "primary_category": "...",
  "secondary_category": "...",
  "confidence": 0.79,
},
{
  "date": "2024-12-02",
  "amount": 2.00,
  "description": "transaction 2",
  "primary_category": "...",
  "secondary_category": "...",
  "confidence": 0.83,
},
{
  "date": "2024-12-03",
  "amount": 3.00,
  "description": "transaction 3",
  "primary_category": "...",
  "secondary_category": "...",
  "confidence": 0.71,
}
```

Run the script with the following command:

```console
python -m financial_tracker \
  --data-dir data \
  --examples-path data/examples.txt
```

The script will concatenate the statements and save it to
`data/statements.pdf`, the metadata to `data/metadata.json`, the extracted
text from the statements to `data/pdf_text.txt`, and the extracted
transactions to `data/transactions.json`.
