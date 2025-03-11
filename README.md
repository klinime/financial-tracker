
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
PDF_SERVICES_CLIENT_ID=...
PDF_SERVICES_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

Refer to
[Adobe PDF Services API](https://developer.adobe.com/document-services/docs/overview/pdf-services-api/gettingstarted/#python)
and [OpenAI API Platform](https://openai.com/api/) for instructions to obtain
the API keys.

PDF services is used in the `extract` stage and OpenAI in the `analyze` stage
(see more details below), where PDF services has a free monthly quota and
OpenAI is pay as you go (cost is logged for each usage). It is recommended
to validate the outputs from the previous stages before executing the above two
to minimize unexpected quota usage or cost.

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
python -m financial_tracker multistage \
  --stages concat extract preprocess analyze correct \
  --data-dirs data-01 data-02 data-03 \
  --examples-path data/examples.txt
```

For each data directory, the script takes the statements as inputs and outputs:

- the concatenated statements to `data/{category}_statements.pdf`
- the metadata to `data/{category}_metadata.json`
- the raw extracted data to `data/{category}_extract/`
- the processed extracted text to `data/{category}_pdf_text.txt`
- the extracted transactions to `data/{category}_transactions.json`
- the corrected transactions to `data/{category}_transactions_corrected.json`

To visualize the transactions in interactive table and charts in a browser:

```console
python -m financial_tracker visualize --data-dirs data-01 data-02 data-03
```

which aggregates all transactions from the specified directories.
