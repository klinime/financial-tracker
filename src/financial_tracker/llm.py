import os
from logging import getLogger

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


class FinancialTransaction(BaseModel):  # type: ignore
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    amount: float = Field(..., description="Transaction amount")
    description: str = Field(..., description="Cleaned transaction description")
    primary_category: str = Field(
        ...,
        description="Primary category of the transaction",
    )
    secondary_category: str = Field(
        ...,
        description="Secondary category of the transaction",
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the primary and secondary categories",
    )

    # @field_validator("primary_category")  # type: ignore
    # def validate_category(cls, v: str) -> str:
    #     valid_primary = {
    #         "income",
    #         "deduction",
    #         "tax",
    #         "saving",
    #         "investment",
    #         "insurance",
    #         "housing",
    #         "utilities",
    #         "transportation",
    #         "food",
    #         "healthcare",
    #         "personal development",
    #         "shopping",
    #         "entertainment",
    #     }
    #     if v not in valid_primary:
    #         raise ValueError(f"Invalid primary category: {v}")
    #     return v

    @field_validator("confidence")  # type: ignore
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class TransactionResponse(BaseModel):  # type: ignore
    transactions: list[FinancialTransaction] = Field(
        ...,
        description="List of extracted financial transactions",
    )


system_prompt = """You are a financial expert specializing in transaction categorization, analysis, and organization.
You are given many statements from multiple financial institutions concatenated together in the <statements> section.
Your task is to extract all the transactions from the statements, extract the relevant information,
and categorize them into primary and secondary categories with confidence scores. The transactions include paychecks, taxes,
deductions, restricted stock units, and more, instead of just transactions from a bank account or credit card.

Follow these rules:

1. Analyze documents with chain-of-thought reasoning:
   a. Identify all monetary transactions
   b. Identify the date, amount, and description of the transaction using surrounding context if necessary
   c. Identify the primary and secondary categories of the transaction from the description or surrounding context
2. Extract the transactions from top to bottom, following the order of the transactions in the statements
3. NEVER skip a transaction - I rather include an uncertain transaction than miss one
4. DO NOT hallucinate transactions - each one must be grounded in the statements
5. DO NOT hallucinate dates - set the date to "--" if you are absolutely cannot identify the date
6. The list of available primary categories are:
   - income
   - deduction
   - tax
   - saving
   - investment
   - insurance
   - housing
   - utilities
   - transportation
   - food
   - healthcare
   - personal development
   - shopping
   - entertainment
6. NEVER guess for primary categories - use confidence scores to express uncertainty
7. Feel free to come up with new categories for secondary categories as needed but try to keep the number of unique secondary categories
   per primary category to a minimum, 10 or less
8. Confidence scores are between 0 and 1, where 0 is "this is a complete guess" and 1 is "this is absolutely certain".
9. Ignore "YTD" or "Employer" information, as it is not relevant to the transactions. Look for "Current" or "Employee" instead.

Use the following to help guide your reasoning:
- The relevant information for each transaction is presented in json-like format as much as possible.
- If a json-like format contains transaction information, the entire json-like format is likely relevant.
- If a json-like format does not contain transaction information, it is likely irrelevant, ignore it.
- The json-like format is not always present for all transactions, in that case try to determine the
  relevancy and categorize them accordingly.
- The json-like format can sometimes be presented in a ill-formed way despite it containing relevant information,
  in that case reason about the information and categorize the transaction accordingly.
- The relevant information for the transactions may not all be present in the json-like format, in that case
  deduce the information from the surrounding context.
- Look for the "Parsed table" and "Raw table" strings to locate possibly relevant tables and context.
- Use the <doc> and <page> tags to locate the relevant information for the transactions.
  The information within these tags are from the same statement or page and could share relevant information, e.g. transaction dates.

<examples>
{examples}
</examples>"""

user_prompt = """
Process these financial documents:
<statements>
{documents}
</statements>

The extracted transactions are:
"""


class FinancialLLM:
    def __init__(self, example_path: str):
        self.logger = getLogger(__name__)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            default_headers={"OpenAI-Beta": "assistants=v2"},
        )
        with open(example_path) as f:
            self.system_prompt = system_prompt.format(examples=f.read())
        self.user_prompt = user_prompt

    def generate_text(self, statements: str) -> str:
        try:
            model = "gpt-4o-2024-08-06"
            self.logger.info(f"Analyzing statements with {model}...")
            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self.user_prompt.format(documents=statements),
                    },
                ],
                response_format=TransactionResponse,
                max_completion_tokens=8192,
                temperature=0.2,
            )

            cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            input_tokens = response.usage.prompt_tokens - cached_tokens
            cached_cost = cached_tokens * 1.25 / 1_000_000
            input_cost = input_tokens * 2.50 / 1_000_000
            output_cost = response.usage.completion_tokens * 10.00 / 1_000_000

            self.logger.info(f"Cached tokens: {cached_tokens}")
            self.logger.info(f"Input tokens: {input_tokens}")
            self.logger.info(f"Output tokens: {response.usage.completion_tokens}")
            self.logger.info(
                f"Cost: ${round(cached_cost + input_cost + output_cost, 6)}"
            )
            output = response.choices[0].message.content
            assert isinstance(output, str)
            return output
        except Exception as e:
            self.logger.error(str(e))
            raise
