def statement_categories() -> list[str]:
    return ["income", "bank", "expense"]


def get_top_level_categories() -> dict[str, str]:
    return {
        "income": "income",
        "deduction": "witholding",
        "tax": "witholding",
        "housing": "necessary expense",
        "utilities": "necessary expense",
        "transportation": "necessary expense",
        "food": "necessary expense",
        "healthcare": "necessary expense",
        "shopping": "discretionary expense",
        "entertainment": "discretionary expense",
        "personal development": "discretionary expense",
    }


def get_primary_categories() -> list[str]:
    return list(get_top_level_categories().keys())
