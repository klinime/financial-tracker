import json
from collections import Counter
from datetime import datetime
from logging import getLogger
from typing import Any

from financial_tracker.categories import get_primary_categories


class TransactionCorrector:
    def __init__(self, transaction_path: str, statement_category: str) -> None:
        self.logger = getLogger(__name__)
        self.transactions: list[dict[str, Any]] = json.load(open(transaction_path))[
            "transactions"
        ]
        months = [
            datetime.strptime(t["date"], "%Y-%m-%d").strftime("%Y-%m")
            for t in self.transactions
        ]
        target_month = Counter(months).most_common(1)[0][0]
        self.transactions = sorted(
            [
                t
                for t in self.transactions
                if datetime.strptime(t["date"], "%Y-%m-%d").strftime("%Y-%m")
                == target_month
            ],
            key=lambda t: t["date"],
        )
        self.statement_category = statement_category
        self.primary_categories = get_primary_categories()
        self.category_mapping = self.build_category_mapping()

    def build_category_mapping(self) -> dict[str, set[str]]:
        category_mapping: dict[str, set[str]] = {}
        for transaction in self.transactions:
            if transaction["primary_category"] not in self.primary_categories:
                continue
            if transaction["primary_category"] not in category_mapping:
                category_mapping[transaction["primary_category"]] = set()
            category_mapping[transaction["primary_category"]].add(
                transaction["secondary_category"]
            )
        return category_mapping

    def inquire_yes_no(self, question: str) -> bool:
        response = input(question).strip().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            self.logger.info("\nInvalid response. Please enter 'y' or 'n'.")
            return self.inquire_yes_no(question)

    def inquire_primary_category(self, transaction: dict[str, Any]) -> str:
        primary_category = (
            input(f"Enter primary category ({transaction['primary_category']}): ")
            .strip()
            .lower()
        )
        if primary_category == "":
            assert isinstance(transaction["primary_category"], str)
            return transaction["primary_category"]
        if primary_category not in self.primary_categories:
            self.logger.info(
                f"\nInvalid primary category. Must be one of: {list(self.primary_categories)}"
            )
            primary_category = self.inquire_primary_category(transaction)
        return primary_category

    def correct_transactions(self) -> list[dict[str, Any]]:
        to_remove: list[dict[str, Any]] = []
        for transaction in self.transactions:
            if (
                (
                    self.statement_category == "bank"
                    and transaction["primary_category"] == "income"
                )
                or (
                    self.statement_category == "bank"
                    and transaction["secondary_category"] == "credit card"
                )
                or (self.statement_category == "expense" and transaction["amount"] < 0)
            ):
                self.logger.info(f"\n\nTransaction: {transaction}")
                remove_transaction = self.inquire_yes_no("Remove transaction? (y/n): ")
                if remove_transaction:
                    to_remove.append(transaction)
            elif (
                transaction["primary_category"] not in self.primary_categories
                or transaction["confidence"] <= 0.5
            ):
                self.logger.info(f"\n\nTransaction: {transaction}")
                self.logger.info(
                    f"\nAvailable primary categories: {list(self.primary_categories)}"
                )
                primary_category = self.inquire_primary_category(transaction)
                if primary_category in self.category_mapping:
                    self.logger.info(
                        f"\nExisting secondary categories for {primary_category}: {list(self.category_mapping[primary_category])}"
                    )
                secondary_category = (
                    input(
                        f"Enter secondary category ({transaction['secondary_category']}): "
                    )
                    .strip()
                    .lower()
                )
                if secondary_category == "":
                    secondary_category = transaction["secondary_category"]
                transaction["primary_category"] = primary_category
                transaction["secondary_category"] = secondary_category
                if primary_category not in self.category_mapping:
                    self.category_mapping[primary_category] = set()
                self.category_mapping[primary_category].add(secondary_category)
        for transaction in to_remove:
            self.transactions.remove(transaction)
        return self.transactions
