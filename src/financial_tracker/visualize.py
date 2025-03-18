import colorsys
import time
from collections.abc import Callable
from functools import wraps
from logging import getLogger
from typing import Any

import dash
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta  # type: ignore

from financial_tracker.categories import get_top_level_categories


class TransactionVisualizer:
    def __init__(self, transactions: list[list[dict[str, Any]]]) -> None:
        self.logger = getLogger(__name__)
        self.category_path = [
            "top_level_category",
            "primary_category",
            "secondary_category",
        ]
        self.top_level_categories = list(
            dict.fromkeys(get_top_level_categories().values())
        )
        self.inflow, self.outflow = self.split_inflow_outflow(
            [
                [
                    self.process_transactions(pd.json_normalize(t))
                    for t in period_transactions
                ]
                for period_transactions in transactions
            ]
        )
        self.transactions = (
            pd.concat(
                [self.inflow, self.outflow.assign(amount=-self.outflow["amount"])]
            )
            .sort_values(
                self.category_path,
                key=lambda x: pd.Categorical(
                    x,
                    categories=self.top_level_categories,
                    ordered=True,
                ),
                kind="stable",
            )
            .reset_index(drop=True)
        )
        self.filtered_transactions = self.transactions
        self.hierarchical_data = self.transactions

    def process_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        transactions["date"] = pd.to_datetime(transactions["date"])
        transactions["amount"] = transactions["amount"].astype(float)
        transactions.insert(
            loc=transactions.columns.get_loc("primary_category"),
            column="top_level_category",
            value=transactions["primary_category"].map(get_top_level_categories()),
        )
        transactions.drop(columns=["confidence"], inplace=True)
        return transactions

    def split_inflow_outflow(
        self, transactions: list[list[pd.DataFrame]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        income_df_list, bank_df_list, expense_df_list = zip(*transactions)
        income_transactions = pd.concat(income_df_list).reset_index(drop=True)
        bank_transactions = pd.concat(bank_df_list).reset_index(drop=True)
        expense_transactions = pd.concat(expense_df_list).reset_index(drop=True)

        income = income_transactions[
            income_transactions["top_level_category"] == "income"
        ]
        witholding = income_transactions[
            income_transactions["top_level_category"] == "witholding"
        ]
        negative_witholding = witholding[witholding["amount"] < 0].copy()
        witholding = witholding[witholding["amount"] >= 0]

        bank_income = bank_transactions[
            bank_transactions["top_level_category"] == "income"
        ]
        bank_expense = bank_transactions[
            bank_transactions["top_level_category"] != "income"
        ]

        income = (
            pd.concat([income, negative_witholding, bank_income])
            .sort_values("date")
            .reset_index(drop=True)
        )
        expense = (
            pd.concat([witholding, bank_expense, expense_transactions])
            .sort_values("date")
            .reset_index(drop=True)
        )
        income["amount"] = income["amount"].abs()
        expense["amount"] = expense["amount"].abs()
        return income, expense

    def build_app(self) -> None:
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.layout = dash.html.Div(
            [
                dash.html.H1(
                    "Financial Tracker Dashboard", style={"textAlign": "center"}
                ),
                dash.dcc.DatePickerRange(
                    id="date-picker",
                    min_date_allowed=self.transactions["date"].min().replace(day=1),
                    max_date_allowed=self.transactions["date"].max()
                    + relativedelta(day=1, months=+1, days=-1),
                    start_date=self.transactions["date"].min(),
                    end_date=self.transactions["date"].max(),
                ),
                dash.html.Div(
                    [
                        dash.html.H3(
                            "Transaction Table", style={"marginBottom": "10px"}
                        ),
                        self.create_data_table(),
                    ],
                    style={"marginLeft": "auto", "marginRight": "auto", "width": "70%"},
                ),
                dash.html.Div(
                    [
                        dash.dcc.Graph(
                            id="treemap-plot",
                            style={"height": 800, "width": "100%"},
                        ),
                        dash.dcc.Graph(
                            id="running-total-plot",
                            style={"height": 400, "width": "100%"},
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr",
                        "gap": "20px",
                    },
                ),
            ]
        )

        self.register_callbacks()

    def register_callbacks(self) -> None:
        def get_visible_data(
            toggle_row: dict[str, Any] | None = None
        ) -> list[dict[str, Any]]:
            if toggle_row:
                row_id = self.hierarchical_data["id"] == toggle_row["id"]
                if toggle_row["is_expanded"]:
                    category = self.category_path[toggle_row["level"]]
                    self.hierarchical_data.loc[
                        self.hierarchical_data[category] == toggle_row[category],
                        ["is_expanded", "display"],
                    ] = False
                    self.hierarchical_data.loc[row_id, "display"] = True
                else:
                    self.hierarchical_data.loc[
                        self.hierarchical_data["parent"] == toggle_row["id"], "display"
                    ] = True
                    self.hierarchical_data.loc[row_id, "is_expanded"] = True

            visible_data: list[dict[str, Any]] = self.hierarchical_data[
                self.hierarchical_data["display"]
            ].to_dict("records")
            for row in visible_data[:-1]:
                if row["level"] < len(self.category_path):
                    # category rows, add expansion icon
                    icon = "▼" if row["is_expanded"] else "▶"
                    display_text = row[self.category_path[row["level"]]]
                    row["category_display"] = f"{icon} {display_text}"
                else:
                    # transaction rows, use date and description
                    row["category_display"] = (
                        f'{row["date"].strftime("%Y-%m-%d")} {row["description"]}'
                    )
            visible_data[-1]["category_display"] = "total"
            return visible_data

        def callback_telemetry(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                func_name = f"{func.__name__}"
                ctx = dash.callback_context
                if not ctx.triggered:
                    self.logger.debug(f"[{func_name}] initial call")
                else:
                    self.logger.debug(f"[{func_name}] triggered by:")
                    for key, value in ctx.triggered[0].items():
                        self.logger.debug(f"[{func_name}] {' ' * 4}{key}: {value}")
                if ctx.inputs:
                    self.logger.debug(f"[{func_name}] inputs:")
                    for key, value in ctx.inputs.items():
                        self.logger.debug(f"[{func_name}] {' ' * 4}{key}: {value}")

                try:
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self.logger.debug(f"[{func_name}] time: {execution_time:.4f}s")
                    return result
                except Exception as e:
                    self.logger.exception(f"[{func_name}] error: {e}")
                    raise

            return wrapper

        @self.app.callback(
            [
                dash.Output("transaction-table", "rowData"),
                dash.Output("treemap-plot", "figure"),
                dash.Output("running-total-plot", "figure"),
            ],
            [
                dash.Input("date-picker", "start_date"),
                dash.Input("date-picker", "end_date"),
            ],
            prevent_initial_call=False,
        )  # type: ignore
        @callback_telemetry
        def update_dashboard(
            start_date: str, end_date: str
        ) -> tuple[list[dict[str, Any]], go.Figure, go.Figure]:
            self.filtered_transactions = self.transactions[
                self.transactions["date"].isin(
                    pd.date_range(start=start_date, end=end_date)
                )
            ]
            self.hierarchical_data = self.create_hierarchical_data()

            visible_data = get_visible_data()

            filtered_inflow = self.inflow[
                self.inflow["date"].isin(pd.date_range(start=start_date, end=end_date))
            ]
            filtered_outflow = self.outflow[
                self.outflow["date"].isin(pd.date_range(start=start_date, end=end_date))
            ]
            sankey_plot = self.create_sankey_plot(filtered_inflow, filtered_outflow)

            line_plot = px.line(
                self.filtered_transactions.sort_values("date"),
                x="date",
                y=self.filtered_transactions.sort_values("date")["amount"].cumsum(),
                title="Cumulative Trend",
            )
            return visible_data, sankey_plot, line_plot

        @self.app.callback(
            dash.Output("transaction-table", "rowData", allow_duplicate=True),
            [dash.Input("transaction-table", "cellClicked")],
            [dash.State("transaction-table", "rowData")],
            prevent_initial_call=True,
        )  # type: ignore
        @callback_telemetry
        def expand_collapse_row(
            cell_clicked: dict[str, Any], data: list[dict[str, Any]]
        ) -> list[dict[str, Any]] | Any:
            if not cell_clicked:
                return dash.no_update
            if cell_clicked["colId"] != "category_display":
                return dash.no_update
            if cell_clicked["value"] == "total":
                return dash.no_update
            clicked_row = data[cell_clicked["rowIndex"]]
            if clicked_row["level"] == len(self.category_path):
                return dash.no_update
            return get_visible_data(clicked_row)

        # @self.app.callback(
        #     dash.Output("transaction-table", "rowData", allow_duplicate=True),
        #     [dash.Input("transaction-table", "columnState")],
        #     [dash.State("transaction-table", "rowData")],
        #     prevent_initial_call=True,
        # )  # type: ignore
        # @callback_telemetry
        # def sort_table(
        #     column_state: list[dict[str, str]], data: list[dict[str, Any]]
        # ) -> list[dict[str, Any]]:
        #     if not column_state:
        #         return data
        #     if not any(state["sort"] for state in column_state):
        #         return dash.no_update

        #     df = pd.DataFrame(data)
        #     # find contiguous transaction blocks
        #     is_transaction = df["level"] == len(self.category_path)
        #     block_starts = is_transaction & ~is_transaction.shift(1).astype(bool)
        #     start_indices = block_starts[block_starts].index
        #     block_ends = is_transaction & ~is_transaction.shift(-1).astype(bool)
        #     end_indices = block_ends[block_ends].index

        #     # sort each transaction block
        #     for start, end in zip(start_indices, end_indices):
        #         block = df.iloc[start : end + 1]
        #         # apply all sort conditions
        #         for state in column_state:
        #             column_id = state["colId"]
        #             is_ascending = state["sort"] == "asc"
        #             if column_id == "amount" or column_id == "percent_total":
        #                 block = block.sort_values(
        #                     column_id, ascending=is_ascending, key=abs
        #                 )
        #             else:
        #                 block = block.sort_values(column_id, ascending=is_ascending)
        #         # replace original block with sorted block
        #         df.iloc[start : end + 1] = block.values
        #     table: list[dict[str, Any]] = df.to_dict("records")
        #     return table

    def create_hierarchical_data(self) -> pd.DataFrame:
        df = self.filtered_transactions.copy()
        inflow_idx = df["amount"] >= 0
        inflow = df[inflow_idx]
        outflow = df[~inflow_idx]
        df.loc[inflow_idx, "percent_in/outflow"] = (
            inflow["amount"] / inflow["amount"].sum()
        )
        df.loc[~inflow_idx, "percent_in/outflow"] = (
            outflow["amount"] / outflow["amount"].sum()
        )
        df["percent_total"] = df["amount"] / df["amount"].sum()

        fields = ["date", "description"]
        values = ["amount", "percent_in/outflow", "percent_total"]

        # populate subtotals for each category
        def populate_subtotals(
            rows: list[dict[str, Any]], cat: str, group: pd.DataFrame, path: list[str]
        ) -> None:
            # all columns: [id, parent] + categories + fields + values + [level, is_expanded, display]
            path = path + [cat]
            row: dict[str, Any] = {
                "id": "_".join(path),
                "parent": "_".join(path[:-1]),
            }
            padded_path = path + [""] * (len(self.category_path) - len(path))
            for c, p in zip(self.category_path, padded_path):
                row[c] = p
            for field in fields:
                row[field] = ""
            for value in values:
                row[value] = group[value].sum()
            row["level"] = len(path) - 1
            row["is_expanded"] = False
            row["display"] = len(path) == 1
            rows.append(row)

            # sort by positive/negative amount then by absolute amount
            if len(path) < len(self.category_path):
                sorted_group = sorted(
                    group.groupby(self.category_path[len(path)], sort=False),
                    key=lambda x: (x[1]["amount"].sum() > 0, abs(x[1]["amount"].sum())),
                    reverse=True,
                )
                for c, g in sorted_group:
                    populate_subtotals(rows, c, g, path)
            else:
                # populate individual transactions
                group = group.sort_values(
                    "amount", key=abs, ascending=False
                ).sort_values("amount", key=lambda x: x > 0, ascending=False)
                for row in group.to_dict("records"):
                    row["id"] = (
                        f"trans_{hash(str(row["date"]) + row["description"] + str(row["amount"]))}"
                    )
                    row["parent"] = "_".join(path)
                    row["level"] = len(path)
                    row["is_expanded"] = False
                    row["display"] = False
                    rows.append(row)

        rows: list[dict[str, Any]] = []
        path: list[str] = []
        sorted_groups = sorted(
            df.groupby("top_level_category", sort=False),
            key=lambda x: self.top_level_categories.index(x[0]),
        )
        for cat, group in sorted_groups:
            populate_subtotals(rows, cat, group, path)
        df = pd.DataFrame(rows)
        total_row = {
            "id": "total",
            "parent": "",
            "top_level_category": "total",
            "primary_category": "",
            "secondary_category": "",
            "amount": df[df["level"] == 0]["amount"].sum(),
            "percent_in/outflow": 1.0,
            "percent_total": 1.0,
            "level": 0,
            "is_expanded": False,
            "display": True,
        }
        for field in fields:
            total_row[field] = ""
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        return df

    def create_data_table(self) -> dag.AgGrid:
        # common_defs = {
        #     "sortable": True,
        #     "comparator": {"function": "(a, b) => 0"},
        # }
        common_defs = {"sortable": False}
        column_defs = [
            {
                "field": "category_display",
                "headerName": "Category",
                "flex": 7,
                "cellStyle": {
                    "styleConditions": [
                        {
                            "condition": f"params.data.level === {level} && params.value !== 'total'",
                            "style": {
                                "paddingLeft": f"{level * 20 + 10}px",
                                "cursor": "pointer",
                            },
                        }
                        for level in range(len(self.category_path) + 1)
                    ]
                },
                **common_defs,
            },
            {
                "field": "amount",
                "headerName": "Amount",
                "type": "numericColumn",
                "valueFormatter": {"function": "d3.format('$,.2f')(params.value)"},
                "flex": 1,
                **common_defs,
            },
            {
                "field": "percent_in/outflow",
                "headerName": r"%inflow or %outflow",
                "type": "numericColumn",
                "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
                "flex": 1,
                **common_defs,
            },
            {
                "field": "percent_total",
                "headerName": r"%total",
                "type": "numericColumn",
                "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
                "flex": 1,
                **common_defs,
            },
        ]
        green_color_gradient = self.color_gradient("#9FE2BF", len(self.category_path))
        red_color_gradient = self.color_gradient("#FAA0A0", len(self.category_path))
        row_style_conditions = [
            {
                "condition": f"params.data.level === {level} && params.data.amount >= 0",
                "style": {
                    "backgroundColor": green_color_gradient[level],
                    "fontWeight": "bold",
                },
            }
            for level in range(len(self.category_path))
        ] + [
            {
                "condition": f"params.data.level === {level} && params.data.amount < 0",
                "style": {
                    "backgroundColor": red_color_gradient[level],
                    "fontWeight": "bold",
                },
            }
            for level in range(len(self.category_path))
        ]
        return dag.AgGrid(
            id="transaction-table",
            columnDefs=column_defs,
            rowData=[],  # populated by callback,
            dashGridOptions={
                "rowSelection": "single",
                "suppressRowClickSelection": False,
                "domLayout": "normal",
                "suppressScrollOnNewData": True,
                "getRowStyle": {"styleConditions": row_style_conditions},
                "animateRows": False,
            },
            style={"height": "500px", "width": "100%"},
        )

    def color_gradient(self, base_color: str, num_levels: int) -> list[str]:
        rgb = tuple(int(base_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(*rgb)
        colors = []
        for i in range(num_levels):
            new_l = l + ((1 - l) * (i / num_levels))
            rgb = colorsys.hls_to_rgb(h, new_l, s)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    def create_sankey_plot(
        self, inflow: pd.DataFrame, outflow: pd.DataFrame
    ) -> go.Figure:
        category_paths = ["primary_category", "secondary_category"]
        intermediary_incomes = ["take home income", "discretionary income"]
        intermediary_expenses = self.top_level_categories[1:]
        assert intermediary_expenses == [
            "witholding",
            "necessary expense",
            "discretionary expense",
        ]
        total_inflow = inflow["amount"].sum()
        total_outflow = outflow["amount"].sum()
        net = total_inflow - total_outflow

        node_id = 0
        nodes = {"total income": node_id}
        node_id += 1
        if net > 0:
            intermediary_incomes.append("surplus")
        elif net < 0:
            intermediary_incomes.append("deficit")
        for income in intermediary_incomes:
            nodes[income] = node_id
            node_id += 1
        for expense in intermediary_expenses:
            nodes[expense] = node_id
            node_id += 1

        percent_node = [1.0] * len(nodes)
        links = []
        percent_link = []
        thicknesses: list[dict[int, float]] = (
            [{} for _ in range(len(category_paths))]
            + [{0: total_inflow}]
            + [{} for _ in range(len(intermediary_expenses) + len(category_paths))]
        )
        blue = "#87CEFA"
        green = "#9FE2BF"
        red = "#FAA0A0"
        colors = (
            [blue]
            + [green] * len(intermediary_incomes)
            + [red] * len(intermediary_expenses)
        )
        if net < 0:
            colors[len(intermediary_incomes)] = red
        # nodes are colors are fully initialized
        # percent_node and thicknesses are partially initialized
        # links and percent_link are not initialized

        def populate_inflow_nodes(
            income_name: str, income_group: pd.DataFrame, depth: int, target_depth: int
        ) -> None:
            nonlocal node_id
            if depth == target_depth:
                nodes[income_name] = node_id
                sum_amount = income_group["amount"].sum()
                percent_node.append(sum_amount / total_inflow)
                thicknesses[len(category_paths) - depth][node_id] = sum_amount
                colors.append(blue)
                if depth == 1:
                    links.append([node_id, 0, sum_amount])
                    percent_link.append(sum_amount / total_inflow)
                else:
                    target_node = nodes[income_group[category_paths[depth - 2]].iloc[0]]
                    links.append([node_id, target_node, sum_amount])
                    percent_link.append(
                        sum_amount / total_inflow / percent_node[target_node]
                    )
                node_id += 1
                return
            sorted_group = sorted(
                income_group.groupby(category_paths[depth], sort=False),
                key=lambda x: x[1]["amount"].sum(),
                reverse=True,
            )
            for income_name, income_group in sorted_group:
                populate_inflow_nodes(
                    income_name, income_group, depth + 1, target_depth
                )

        def populate_outflow_categories() -> None:
            remaining_income = total_inflow
            for depth, expense_category in enumerate(intermediary_expenses):
                sub_expense = outflow[outflow["top_level_category"] == expense_category]
                subtotal = sub_expense["amount"].sum()
                remaining_income -= subtotal

                income_id = depth + 1
                percent_node[income_id] = abs(remaining_income) / total_inflow
                income_level = len(category_paths) + 1 + depth
                if remaining_income > 0:
                    # surplus branches from discretionary income
                    thicknesses[income_level][income_id] = remaining_income
                    links.append([income_id - 1, income_id, remaining_income])
                    percent_link.append(
                        remaining_income / total_inflow / percent_node[income_id - 1]
                    )

                expense_id = nodes[expense_category]
                percent_node[expense_id] = subtotal / total_inflow
                thicknesses[len(category_paths) + 1 + depth][expense_id] = subtotal
                link_amount = min(remaining_income, subtotal)
                links.append([income_id - 1, expense_id, link_amount])
                percent_link.append(
                    link_amount / total_inflow / percent_node[income_id - 1]
                )
                if remaining_income < 0:
                    # deficit branches into discretionary expense
                    thicknesses[income_level - 1][income_id] = -remaining_income
                    links.append([income_id, expense_id, -remaining_income])
                    percent_link.append(-remaining_income / subtotal)

        def populate_outflow_nodes(
            expense_name: str,
            expense_group: pd.DataFrame,
            depth: int,
            target_depth: int,
            leaf_expense_ids: list[int],
        ) -> None:
            nonlocal node_id
            if depth == target_depth:
                if depth == len(category_paths):
                    leaf_expense_ids.append(node_id)
                if expense_name in nodes:
                    nodes[f"{expense_name} income"] = nodes.pop(expense_name)
                    expense_name = f"{expense_name} expense"
                nodes[expense_name] = node_id
                sum_amount = expense_group["amount"].sum()
                percent_node.append(sum_amount / total_inflow)
                category = expense_group["top_level_category"].iloc[0]
                thicknesses[
                    len(category_paths)
                    + 1
                    + intermediary_expenses.index(category)
                    + depth
                ][node_id] = sum_amount
                colors.append(red)
                source_category = category
                if depth > 1:
                    source_category = expense_group[category_paths[depth - 2]].iloc[0]
                    if source_category not in nodes:
                        source_category = f"{source_category} expense"
                links.append([nodes[source_category], node_id, sum_amount])
                percent_link.append(
                    sum_amount / total_inflow / percent_node[nodes[source_category]]
                )
                node_id += 1
                return

            sorted_group = sorted(
                expense_group.groupby(category_paths[depth], sort=False),
                key=lambda x: x[1]["amount"].sum(),
                reverse=True,
            )
            for expense_name, expense_group in sorted_group:
                populate_outflow_nodes(
                    expense_name,
                    expense_group,
                    depth + 1,
                    target_depth,
                    leaf_expense_ids,
                )

        leaf_expense_ids: list[int] = []
        for i in range(len(category_paths)):
            populate_inflow_nodes(category_paths[i], inflow, 0, i + 1)
        populate_outflow_categories()
        for i in range(len(category_paths)):
            populate_outflow_nodes(
                category_paths[i], outflow, 0, i + 1, leaf_expense_ids
            )

        labels = [k for k, _ in sorted(nodes.items(), key=lambda x: x[1])]
        x_positions, y_positions = self.calculate_node_positions(
            thicknesses,
            anchor_id=0,
            anchor_xpos=0.4,
            leaf_expense_ids=leaf_expense_ids,
        )
        source, target, value = zip(*links)
        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=labels,
                    x=x_positions,
                    y=y_positions,
                    color=colors,
                    customdata=percent_node,
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    hovertemplate="$%{value:,.2f}<br>%{customdata:,.2%} of total income<extra></extra>",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    customdata=percent_link,
                    hovertemplate="$%{value:,.2f}<extra>%{customdata:,.2%}</extra>",
                ),
            )
        )
        fig.update_layout(title="Financial Overview")
        return fig

    def calculate_node_positions(
        self,
        column_thicknesses: list[dict[int, float]],
        anchor_id: int = 0,
        anchor_xpos: float = 0.5,
        leaf_expense_ids: list[int] | None = None,
    ) -> tuple[list[float], list[float]]:
        x_positions = [0.0] * sum(len(col) for col in column_thicknesses)
        y_positions = [0.5] * len(x_positions)
        anchor_col = 0
        for i, col in enumerate(column_thicknesses):
            if anchor_id in col and len(col) == 1:
                anchor_col = i
                max_thickness = col[anchor_id]
                break
        left_x_offset = anchor_xpos / anchor_col
        right_x_offset = (1.0 - anchor_xpos) / (
            len(column_thicknesses) - anchor_col - 1
        )

        base_spacing = 0.1
        min_spacing = 0.02
        band_width = 0.6
        for idx, col in enumerate(column_thicknesses):
            x_pos = (
                left_x_offset * idx
                if idx < anchor_col
                else anchor_xpos + right_x_offset * (idx - anchor_col)
            )
            for node_id in col.keys():
                x_positions[node_id] = x_pos

            num_min_spacing = (
                sum(1 for node_id in col if node_id in leaf_expense_ids)
                if leaf_expense_ids
                else 0
            )
            num_std_spacing = len(col) - max(1, num_min_spacing)
            factor = abs(anchor_id - idx)
            spacing = 0.0 if factor == 0 else base_spacing / factor
            margin = band_width * sum(col.values()) / max_thickness
            while True:
                available_space = 1.0 - (
                    margin + min_spacing * num_min_spacing + spacing * num_std_spacing
                )
                if available_space > 0:
                    break
                factor += 1
                spacing = base_spacing / factor
            y_pos = available_space / 2
            for node_id, thickness in col.items():
                half_width = thickness / max_thickness * band_width / 2
                y_pos += half_width
                y_positions[node_id] = y_pos
                y_pos += half_width
                if leaf_expense_ids and node_id in leaf_expense_ids:
                    y_pos += min_spacing
                else:
                    y_pos += spacing
        return x_positions, y_positions

    def run(self) -> None:
        self.app.run_server(debug=True)
