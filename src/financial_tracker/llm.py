from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class FinancialTransaction(BaseModel):  # type: ignore
    transaction_id: str = Field(..., description="Unique transaction identifier")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    amount: float = Field(..., description="Transaction amount")
    description: str = Field(..., description="Cleaned transaction description")
    category: dict[str, str | float] = Field(
        ...,
        description="""Nested categorization with confidence scores:
        - primary: income, deduction, tax, saving, investment, housing, utilities, transportation, food, healthcare, personal development, entertainment
        - secondary: payroll, 401(k), federal, mortgage, water, public transport, groceries, electronics, etc.
        - confidence: 0.0-1.0""",
    )
    cross_references: list[str] = Field(
        default_factory=list, description="Linked transaction IDs from other documents"
    )


system_prompt = """You are a financial expert specializing in transaction categorization, analysis, and organization.
You are given many statements from multiple financial institutions concatenated together in the [STATEMENTS] section.
Your task is to extract all the transactions from the statements, extract the relevant information,
and categorize them into primary and secondary categories with confidence scores. The transactions include paychecks, taxes,
deductions, restricted stock units, and more, instead of just transactions from a bank account or credit card.

Follow these rules:

1. Analyze documents with chain-of-thought reasoning:
   a. Identify all monetary transactions
   b. Identify the date, amount, and description of the transaction using surrounding context if necessary
   c. Identify the primary and secondary categories of the transaction
   d. Identify related transactions across documents for internal transfers where one transaction is the source and one or more
      transactions are the destination(s)
   e. Identify the cross-reference transaction IDs only for the transactions that are the destination(s) of an internal transfer
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
   - housing
   - utilities
   - transportation
   - food
   - healthcare
   - personal development
   - entertainment
6. NEVER guess for primary categories - use confidence scores to express uncertainty
7. Feel free to come up with new categories for secondary categories as needed but try to keep the number of unique secondary categories
   per primary category to a minimum, 10 or less
8. Ignore "YTD" or "Employer" information, as it is not relevant to the transactions. Look for "Current" or "Employee" instead.

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
- Use the <doc>, </doc> tags and <page>, </page> tags to locate the relevant information for the transactions.
  The information within these tags are from the same statement or page and could share relevant information, e.g. transaction dates.

{format_instructions}"""


class FinancialLLM:
    def __init__(self, example_path: str):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """
Process these financial documents:
[EXAMPLES]
{examples}

[STATEMENTS]
{documents}

[OUTPUT]
The extracted transactions are:
""",
                ),
            ]
        )
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=4096,
            model_kwargs={
                "response_format": {"type": "json_object"},
                "seed": 42,  # For deterministic output
            },
        )
        self.parser = JsonOutputParser(pydantic_object=FinancialTransaction)
        self.processing_chain = self.prompt | self.model | self.parser
        with open(example_path) as f:
            self.examples = f.read()
        self.cost = 0.0

    def generate_text(self, statements: str) -> str:
        with get_openai_callback() as cb:
            result = self.processing_chain.invoke(
                {
                    "format_instructions": self.parser.get_format_instructions(),
                    "examples": self.examples,
                    "documents": statements,
                }
            )
            self.cost = cb.total_cost
            assert isinstance(result, str)
            return result

    def get_cost(self) -> float:
        return self.cost
