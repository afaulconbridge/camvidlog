from typing import Any

import shiny

# Part 1: ui ----
app_ui = shiny.ui.page_fluid(
    shiny.ui.panel_title("CamVidLog"),
    shiny.ui.output_text("result"),
)


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002
    @output
    @shiny.render.text
    def result():
        return "hello world"


# Combine into a shiny app.
# Note that the variable must be "app".
app = shiny.App(app_ui, server)
