import base64
from pathlib import Path
from typing import Any

import shiny

from camvidlog.config import ConfigService
from camvidlog.db.service import DbService

# Part 1: ui ----
app_ui = shiny.ui.page_fluid(
    shiny.ui.panel_title("CamVidLog", "CamVidLog"),
    shiny.ui.output_ui("videos"),
)

config = ConfigService()
db_service = DbService(config.database_url)


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002, ARG001
    @output
    @shiny.render.ui
    def videos():
        records = db_service.get_tracks_with_videos()
        # filename
        # frame from-to
        results = []
        for video, track in records:
            result_parts = [
                shiny.ui.h5(video.filename),
                shiny.ui.tags.video(src=video.filename, width=360, controls=True, muted=True),
            ]
            if track:
                result_parts.extend(
                    (
                        shiny.ui.div(
                            f"{track.frame_first:04d}-{track.frame_last:04d} ({track.frame_last-track.frame_first+1})"
                        ),
                        shiny.ui.div(
                            shiny.ui.img(
                                src=f"data:image/jpeg;base64, {base64.standard_b64encode(track.thumb_first).decode()}"
                            ),
                            shiny.ui.img(
                                src=f"data:image/jpeg;base64, {base64.standard_b64encode(track.thumb_mid).decode()}"
                            ),
                            shiny.ui.img(
                                src=f"data:image/jpeg;base64, {base64.standard_b64encode(track.thumb_last).decode()}"
                            ),
                        ),
                    )
                )

            results.append(shiny.ui.div(*result_parts))

        return shiny.ui.page_fluid(*results)


# Combine into a shiny app.
# hardcode the path to the datafiles for now
www_dir = Path(__file__).parent.parent.parent
# NB: variable must be "app".
app = shiny.App(app_ui, server, static_assets=www_dir)
