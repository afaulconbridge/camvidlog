import base64
import itertools
import math
from pathlib import Path
from typing import Any

import shiny

from camvidlog.config import ConfigService
from camvidlog.db.service import DbService

# Part 1: ui ----
app_ui = shiny.ui.page_fluid(
    shiny.ui.panel_title("CamVidLog", "CamVidLog"),
    shiny.ui.tags.div(
        shiny.ui.input_selectize("page_size", "Page Size", {5: "5", 10: "10", 25: "25", 50: "50"}, selected="5"),
        shiny.ui.input_slider("page_number", "Page Number", min=1, max=2, value=1, step=1),
        shiny.ui.input_select("results_filter", "Result Filter", choices=["none"], multiple=True),
    ),
    shiny.ui.tags.hr(),
    shiny.ui.output_ui("videos"),
)

config = ConfigService()
db_service = DbService(config.database_url)


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002, ARG001
    @shiny.reactive.Calc
    def get_record_count():
        filtered = input.results_filter()
        filtered = [None if i == "none" else i for i in input.results_filter()]
        return len(db_service.get_tracks_with_videos(filtered))

    @shiny.reactive.Calc
    def get_results_seen():
        return db_service.get_track_results()

    @shiny.reactive.Calc
    def get_records():
        page_size = int(input.page_size())
        page_offset = int(input.page_number()) - 1  # minus 1 to go from 1-based UI to 0-based code
        filtered = [None if i == "none" else i for i in input.results_filter()]

        return tuple(
            itertools.islice(
                db_service.get_tracks_with_videos(filtered),
                page_size * page_offset,
                page_size * (page_offset + 1),
            )
        )

    @shiny.reactive.Effect()
    def update_page_number():
        # update number of pages based on page size and number of records
        num_pages = (get_record_count() // int(input.page_size())) + 1  # plus 1 to go from 0-based code to 1-based UI
        shiny.ui.update_slider("page_number", max=num_pages)

    @shiny.reactive.Effect()
    def update_filter():
        # update filter options
        shiny.ui.update_select("results_filter", choices=[*list(get_results_seen()), "none"])

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
                # TODO get FPS from stored metadata
                fps = 30
                first_min = track.frame_first // (fps * 60)
                first_sec = (track.frame_first // fps) % 60
                last_min = track.frame_last // (fps * 60)
                last_sec = (track.frame_last // fps) % 60
                first_time = f"{first_min:02d}:{first_sec:02d}"
                last_time = f"{last_min:02d}:{last_sec:02d}"
                result_parts.extend(
                    (
                        shiny.ui.div(
                            shiny.ui.p(
                                f'{first_time}-{last_time} ({track.frame_last-track.frame_first+1}f) "{track.result}" ({track.result_score:.3f})'
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
