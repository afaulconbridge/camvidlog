from typing import Any

import pandas as pd
import shiny

from camvidlog.config import ConfigService
from camvidlog.db.service import DbService

# Part 1: ui ----
app_ui = shiny.ui.page_fluid(
    shiny.ui.panel_title("CamVidLog"),
    shiny.ui.output_text("result"),
    shiny.ui.output_table("videos"),
)

config = ConfigService()
db_service = DbService(config.database_url)


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002, ARG001
    @output
    @shiny.render.text
    def result():
        return "hello world"

    @output
    @shiny.render.table
    def videos():
        records = db_service.get_tracks_with_videos()
        dicts = []
        for r in records:
            video, track = r
            d = {}
            for k, v in dict(video.model_dump()).items():
                d[f"v.{k}"] = v
            for k, v in dict(track.model_dump()).items():
                d[f"t.{k}"] = v
            dicts.append(d)
        df = pd.DataFrame.from_records(dicts, exclude=("v.id_", "t.video_id", "t.id_"))
        return df


# Combine into a shiny app.
# Note that the variable must be "app".
app = shiny.App(app_ui, server)
