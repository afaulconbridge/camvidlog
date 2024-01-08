import base64
import itertools
from pathlib import Path
from typing import Any

import shiny

from camvidlog.config import ConfigService
from camvidlog.db.service import DbService

# Part 1: ui ----
app_ui = shiny.ui.page_fluid(
    shiny.ui.panel_title("CamVidLog", "CamVidLog"),
    shiny.ui.div(
        shiny.ui.input_selectize("page_size", "Page Size", {5: "5", 10: "10", 25: "25", 50: "50"}, selected="5"),
    ),
    shiny.ui.tags.hr(),
    shiny.ui.output_ui("videos"),
)

config = ConfigService()
db_service = DbService(config.database_url)


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002, ARG001
    @shiny.reactive.Calc
    def get_records():
        page_size = int(input.page_size())
        page_offset = 0
        return tuple(
            itertools.islice(
                db_service.get_tracks_with_videos(),
                page_size * page_offset,
                page_size * (page_offset + 1),
            )
        )

    @output
    @shiny.render.ui
    def videos():
        records = get_records()
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
                            shiny.ui.p(
                                f'{track.frame_first:04d}-{track.frame_last:04d} ({track.frame_last-track.frame_first+1}) "{track.result}" ({track.result_score:.3f})'
                            )
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

            result_parts.append(shiny.ui.tags.hr())
            results.append(shiny.ui.div(*result_parts))

        return shiny.ui.page_fluid(*results)


# Combine into a shiny app.
# hardcode the path to the datafiles for now
www_dir = Path(__file__).parent.parent.parent
# NB: variable must be "app".
app = shiny.App(app_ui, server, static_assets=www_dir)
