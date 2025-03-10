import colorsys
from typing import Any

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta  # type: ignore
from plotly.subplots import make_subplots

from financial_tracker.categories import get_top_level_categories


class TransactionVisualizer:
    def __init__(self, transactions: list[dict[str, Any]]) -> None:
        self.inflow, self.outflow = self.split_inflow_outflow(
            [self.process_transactions(pd.json_normalize(t)) for t in transactions]
        )
        self.category_path = [
            "top_level_category",
            "primary_category",
            "secondary_category",
        ]
        self.top_level_categories = list(
            dict.fromkeys(get_top_level_categories().values())
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
        self.transactions["percent_total"] = (
            self.transactions["amount"] / self.transactions["amount"].sum()
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
        self, transactions: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        income_transactions = transactions[0]
        income = income_transactions[
            income_transactions["top_level_category"] == "income"
        ]
        witholding = income_transactions[
            income_transactions["top_level_category"] == "witholding"
        ]
        negative_witholding = witholding[witholding["amount"] < 0].copy()
        witholding = witholding[witholding["amount"] >= 0]

        bank_transactions = transactions[1]
        bank_income = bank_transactions[
            bank_transactions["top_level_category"] == "income"
        ]
        bank_expense = bank_transactions[
            bank_transactions["top_level_category"] != "income"
        ]

        expense_transactions = transactions[2]
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
        income["percent_in/outflow"] = income["amount"] / income["amount"].sum()
        expense["percent_in/outflow"] = expense["amount"] / expense["amount"].sum()
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

        @self.app.callback(
            [
                dash.Output("transaction-table", "data"),
                dash.Output("treemap-plot", "figure"),
                dash.Output("running-total-plot", "figure"),
            ],
            [
                dash.Input("date-picker", "start_date"),
                dash.Input("date-picker", "end_date"),
            ],
            prevent_initial_call=False,
        )  # type: ignore
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
            treemap = self.create_treemap_plot(filtered_inflow, filtered_outflow)

            line_plot = px.line(
                self.filtered_transactions.sort_values("date"),
                x="date",
                y=self.filtered_transactions.sort_values("date")["amount"].cumsum(),
                title="Cumulative Trend",
            )

            return visible_data, treemap, line_plot

        @self.app.callback(
            dash.Output("transaction-table", "data", allow_duplicate=True),
            [dash.Input("transaction-table", "active_cell")],
            [dash.State("transaction-table", "data")],
            prevent_initial_call=True,
        )  # type: ignore
        def expand_collapse_row(
            active_cell: dict[str, Any], current_data: list[dict[str, Any]]
        ) -> list[dict[str, Any]] | Any:
            if not active_cell:
                return dash.no_update
            if active_cell["column_id"] != "category_display":
                return dash.no_update
            row_idx = active_cell["row"]
            if row_idx >= len(current_data):
                return dash.no_update
            clicked_row = current_data[row_idx]
            if clicked_row["level"] == len(self.category_path):
                return dash.no_update

            return get_visible_data(clicked_row)

        @self.app.callback(
            dash.Output("transaction-table", "data", allow_duplicate=True),
            [dash.Input("transaction-table", "sort_by")],
            [dash.State("transaction-table", "data")],
            prevent_initial_call=True,
        )  # type: ignore
        def sort_table(
            sort_by: list[dict[str, str]], data: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            if not sort_by:
                return data

            df = pd.DataFrame(data)
            # find contiguous transaction blocks
            is_transaction = df["level"] == len(self.category_path)
            block_starts = is_transaction & ~is_transaction.shift(1).astype(bool)
            start_indices = block_starts[block_starts].index
            block_ends = is_transaction & ~is_transaction.shift(-1).astype(bool)
            end_indices = block_ends[block_ends].index

            # sort each transaction block
            for start, end in zip(start_indices, end_indices):
                block = df.iloc[start : end + 1]
                # apply all sort conditions
                for sort_item in sort_by:
                    column_id = sort_item["column_id"]
                    is_ascending = sort_item["direction"] == "asc"
                    if column_id == "amount" or column_id == "percent_total":
                        block = block.sort_values(
                            column_id, ascending=is_ascending, key=abs
                        )
                    else:
                        block = block.sort_values(column_id, ascending=is_ascending)
                # replace original block with sorted block
                df.iloc[start : end + 1] = block.values
            table: list[dict[str, Any]] = df.to_dict("records")
            return table

    def create_hierarchical_data(self) -> pd.DataFrame:
        df = self.filtered_transactions.copy()

        fields = ["date", "description"]
        values = ["amount", "percent_in/outflow", "percent_total"]

        # populate subtotals for each category
        def dfs(
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
                    dfs(rows, c, g, path)
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
            dfs(rows, cat, group, path)
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

    def create_data_table(self) -> dash.dash_table.DataTable:
        columns = [
            {
                "name": "Category",
                "id": "category_display",
                "type": "text",
            },
            {
                "name": "Amount",
                "id": "amount",
                "type": "numeric",
                "format": {"specifier": "$,.2f"},
            },
            {
                "name": r"%inflow or %outflow",
                "id": "percent_in/outflow",
                "type": "numeric",
                "format": {"specifier": ".2%"},
            },
            {
                "name": r"%total",
                "id": "percent_total",
                "type": "numeric",
                "format": {"specifier": ".2%"},
            },
        ]

        style_data_conditional = [
            {
                "if": {"column_id": "category_display"},
                "cursor": "pointer",
                "width": "70%",
            },
            {
                "if": {"column_id": "amount"},
                "width": "10%",
            },
            {
                "if": {"column_id": "percent_in/outflow"},
                "width": "10%",
            },
            {
                "if": {"column_id": "percent_total"},
                "width": "10%",
            },
        ]
        # level-based indentation
        for level, color in enumerate(
            self.color_gradient("#ADD8E6", len(self.category_path))
        ):
            style_data_conditional.append(
                {
                    "if": {"filter_query": f"{{level}} eq {level}"},
                    "backgroundColor": color,
                    "fontWeight": "bold",
                }
            )
            style_data_conditional.append(
                {
                    "if": {
                        "state": "selected",
                        "filter_query": f"{{level}} eq {level}",
                    },
                    "backgroundColor": color,
                    "border": "1px solid rgb(211, 211, 211)",
                }
            )
            style_data_conditional.append(
                {
                    "if": {
                        "column_id": "category_display",
                        "filter_query": f"{{level}} eq {level}",
                    },
                    "paddingLeft": str(level * 20 + 10) + "px",
                }
            )
        style_data_conditional.append(
            {
                "if": {
                    "column_id": "category_display",
                    "filter_query": f"{{level}} eq {len(self.category_path)}",
                },
                "paddingLeft": str(len(self.category_path) * 20 + 10) + "px",
            }
        )

        return dash.dash_table.DataTable(
            id="transaction-table",
            columns=columns,
            data=[],  # populated by callback
            row_selectable=False,
            selected_rows=[],
            style_cell={
                "textAlign": "left",
                "padding": "5px",
                "whiteSpace": "normal",
                "height": "auto",
                "maxWidth": 0,
            },
            style_data_conditional=style_data_conditional,
            style_header={
                "backgroundColor": "rgb(230, 230, 230)",
                "fontWeight": "bold",
            },
            sort_action="custom",
            sort_mode="multi",
            sort_by=[],  # populated by callback
            style_table={
                "overflowX": "auto",
                "overflowY": "auto",
                "height": "500px",
                "width": "100%",
            },
            css=[{"selector": ".dash-spreadsheet", "rule": "width: 100%;"}],
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

    def create_treemap_plot(
        self, inflow: pd.DataFrame, outflow: pd.DataFrame
    ) -> go.Figure:
        display_kwargs = {
            "values": "amount",
            "color": "amount",
        }
        update_kwargs = {
            "branchvalues": "total",
            "hovertemplate": "<b>%{label}</b><br>Amount: $%{value}<br><extra></extra>",
        }

        income_fig = px.treemap(
            inflow,
            path=["primary_category", "secondary_category"],
            title="Income Distribution",
            **display_kwargs,
        )
        income_fig.update_traces(**update_kwargs)
        income_trace = income_fig.data[0]
        income_trace.marker.coloraxis = "coloraxis1"
        income_trace.marker.showscale = True

        total_inflow = inflow["amount"].sum()
        total_outflow = outflow["amount"].sum()
        if total_inflow > total_outflow:
            surplus = pd.DataFrame(
                {
                    "date": [inflow["date"].max()],
                    "amount": [total_inflow - total_outflow],
                    "description": ["surplus"],
                    "top_level_category": ["income"],
                    "primary_category": ["income"],
                    "secondary_category": ["surplus"],
                }
            )
            outflow_with_surplus = pd.concat([outflow, surplus])
            expense_fig = px.treemap(
                outflow_with_surplus,
                path=self.category_path,
                **display_kwargs,
            )
            expense_fig.update_traces(**update_kwargs)
            expense_trace = expense_fig.data[0]
            expense_trace.marker.coloraxis = "coloraxis2"
            expense_trace.marker.showscale = True

            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=[
                    "Income Distribution",
                    "Expense Distribution (with Surplus)",
                ],
                specs=[[{"type": "domain"}], [{"type": "domain"}]],
                row_heights=[0.5, 0.5],
            )
            fig.add_trace(income_trace, row=1, col=1)
            fig.add_trace(expense_trace, row=2, col=1)
        else:
            sorted_outflow = outflow.sort_values(
                "top_level_category",
                key=lambda x: pd.Categorical(
                    x,
                    categories=self.top_level_categories,
                    ordered=True,
                ),
            )
            cumsum = sorted_outflow["amount"].cumsum()
            index = cumsum.searchsorted(total_inflow)
            within_budget = sorted_outflow.iloc[:index]
            over_budget = sorted_outflow.iloc[index:]
            remaining_inflow = total_inflow - within_budget["amount"].sum()
            remaining = pd.DataFrame(
                {
                    "date": [within_budget["date"].max()],
                    "amount": [remaining_inflow],
                    "description": ["remaining income"],
                    "top_level_category": ["income"],
                    "primary_category": ["income"],
                    "secondary_category": ["remaining income"],
                }
            )
            within_budget_with_remaining = pd.concat([within_budget, remaining])

            within_budget_fig = px.treemap(
                within_budget_with_remaining,
                path=self.category_path,
                **display_kwargs,
            )
            within_budget_fig.update_traces(**update_kwargs)
            within_budget_trace = within_budget_fig.data[0]
            within_budget_trace.marker.coloraxis = "coloraxis2"
            within_budget_trace.marker.showscale = False
            over_budget_fig = px.treemap(
                over_budget,
                path=self.category_path,
                **display_kwargs,
            )
            over_budget_fig.update_traces(**update_kwargs)
            over_budget_trace = over_budget_fig.data[0]
            over_budget_trace.marker.coloraxis = "coloraxis2"
            over_budget_trace.marker.showscale = True

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Income Distribution",
                    "Expense Distribution (Within Income)",
                    "Excess Expenses",
                ],
                specs=[
                    [{"type": "domain", "colspan": 2}, None],
                    [{"type": "domain"}, {"type": "domain"}],
                ],
                row_heights=[0.5, 0.5],
                column_widths=[
                    total_inflow / total_outflow,
                    1 - total_inflow / total_outflow,
                ],
            )
            fig.add_trace(income_trace, row=1, col=1)
            fig.add_trace(within_budget_trace, row=2, col=1)
            fig.add_trace(over_budget_trace, row=2, col=2)

        inflow_aggr = inflow.groupby("secondary_category")["amount"].sum()
        fig.update_layout(
            coloraxis1=dict(
                colorscale="Greens",
                cmin=inflow_aggr.min(),
                cmax=inflow_aggr.max(),
                colorbar=dict(
                    title="Amount",
                    xanchor="left",
                    x=1.1,
                    yanchor="top",
                    y=1,
                    len=0.4,
                ),
            ),
        )
        outflow_aggr = (
            (outflow_with_surplus if total_inflow > total_outflow else outflow)
            .groupby("secondary_category")["amount"]
            .sum()
        )
        fig.update_layout(
            coloraxis2=dict(
                colorscale="Reds",
                cmin=outflow_aggr.min(),
                cmax=outflow_aggr.max(),
                colorbar=dict(
                    title="Amount",
                    xanchor="left",
                    x=1.1,
                    yanchor="bottom",
                    y=0,
                    len=0.4,
                ),
            ),
        )
        fig.update_layout(title="Financial Overview")
        return fig

    def run(self) -> None:
        self.app.run_server(debug=True)
