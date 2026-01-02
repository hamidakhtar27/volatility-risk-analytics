import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

app.title = "Volatility & Risk Analytics System"

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div(value, className="kpi-value"),
            ]
        ),
        className="kpi-card",
    )

# -------------------------------------------------
# Layout
# -------------------------------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[

        # ================= HEADER =================
        html.Div(
            [
                html.H1("Volatility & Risk Analytics System", className="header-title"),
                html.Div(
                    "Volatility Modeling, Value-at-Risk & Regulatory Backtesting",
                    className="header-subtitle",
                ),
            ],
            className="header",
        ),

        # ================= KPIs =================
        dbc.Row(
            [
                dbc.Col(kpi_card("ML VaR Breach Rate (99%)", "0.78%"), md=3),
                dbc.Col(kpi_card("Expected Breach Rate", "1.00%"), md=3),
                dbc.Col(kpi_card("Kupiec p-value", "0.117"), md=3),
                dbc.Col(kpi_card("Christoffersen p-value", "0.002"), md=3),
            ],
            className="mb-4",
        ),

        # ================= TABS =================
        dbc.Tabs(
            [

                # -------- Overview / VaR --------
                dbc.Tab(
                    label="📉 VaR Validation",
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "ML Tail-Risk Validation: VaR Breaches (99%)",
                                    className="section-title",
                                ),
                                html.Img(
                                    src="/assets/figures/var_breaches.png",
                                    className="figure-img",
                                ),
                            ],
                            className="section-card",
                        )
                    ],
                ),

                # -------- Basel --------
                dbc.Tab(
                    label="🚦 Basel Traffic Light",
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "Basel Traffic Light — Rolling 250-Day VaR Breaches",
                                    className="section-title",
                                ),
                                html.Img(
                                    src="/assets/figures/basel_traffic_light.png",
                                    className="figure-img",
                                ),
                            ],
                            className="section-card",
                        )
                    ],
                ),

                # -------- Volatility --------
                dbc.Tab(
                    label="📈 Volatility Models",
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "Realized Volatility vs GARCH-family Models",
                                    className="section-title",
                                ),
                                html.Img(
                                    src="/assets/figures/volatility_models.png",
                                    className="figure-img",
                                ),
                            ],
                            className="section-card",
                        )
                    ],
                ),

                # -------- Stress --------
                dbc.Tab(
                    label="⚠️ Stress Testing",
                    children=[
                        html.Div(
                            [
                                html.H4(
                                    "Stress Test — Cumulative Drawdowns",
                                    className="section-title",
                                ),
                                html.Img(
                                    src="/assets/figures/stress_drawdowns.png",
                                    className="figure-img",
                                ),
                            ],
                            className="section-card",
                        )
                    ],
                ),
            ],
            className="mb-4",
        ),

        # ================= FOOTER =================
        html.Footer(
            "Developed by Mohd Hamid Akhtar Khan · Quantitative Risk Analytics",
            className="footer",
        ),
    ],
)

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
