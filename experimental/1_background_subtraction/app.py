import base64
import glob
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import shiny

# Part 1: ui ----
app_ui = shiny.ui.page_sidebar(
    shiny.ui.sidebar(
        shiny.ui.input_slider("downscale", "Downscale factor", min=1, max=16, value=4, step=1),
        shiny.ui.input_slider("knn", "KNN threshold", min=10, max=2000, value=50, step=10),
        shiny.ui.input_slider("mog", "MOG threshold", min=1, max=64, value=4, step=1),
        shiny.ui.input_slider("kernel_open_size", "Kernel size (open)", min=3, max=19, value=5, step=2),
        shiny.ui.input_slider("kernel_close_size", "Kernel size (close)", min=3, max=19, value=7, step=2),
        shiny.ui.input_switch("precalc", "Pre-calculate background", False),
        shiny.ui.input_switch("equalization", "Equalization", False),
        shiny.ui.input_slider("equalization_tiles", "Equalization tiles", min=0, max=32, value=8, step=1),
        shiny.ui.input_slider("equalization_clip", "Equalization clip", min=1, max=100, value=40, step=1),
        shiny.ui.input_slider("frame_number", "Frame Number", min=1, max=900, value=5, step=1, animate=True),
    ),
    shiny.ui.panel_title("CamVidLog Experiments #1", "CamVidLog Experiments #1"),
    shiny.ui.output_ui("result"),
)

filename = random.choice(glob.glob("/workspaces/camvidlog/data/*.MP4"))
# fox walking
# filename = "/workspaces/camvidlog/data/20231121002042_VD_00342.MP4"
# deer eating
# filename = "/workspaces/camvidlog/data/20231201183632_VD_00461.MP4"
# otter eating
# filename = "/workspaces/camvidlog/data/20231128221316_VD_00439.MP4"
# hedgehog
filename = "/workspaces/camvidlog/data/20231124015218_VD_00348.MP4"
filename = "/workspaces/camvidlog/data/20231119233300_VD_00334.MP4"
print(filename)


def downscale_image(image: cv2.typing.MatLike, factor: float) -> cv2.typing.MatLike:
    if factor == 1:
        return image
    else:
        output_size = (image.shape[1] // factor, image.shape[0] // factor)
        return cv2.resize(image, output_size)


def image_to_b64str(image: cv2.typing.MatLike) -> bytes:
    # turn image into jpeg bytes
    img_encode = cv2.imencode(".jpg", image)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    return base64.standard_b64encode(byte_encode).decode()


def get_clusters(frame: cv2.typing.MatLike, no_clusters=1):
    # https://stackoverflow.com/a/70565115/932342

    # Prepare to do some K-means
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    # Find x,y coordinates of all non-black pixels
    z = np.column_stack(np.where(frame == 255)).astype(np.float32)
    if not len(z):
        return (0, 0, frame.shape[1], frame.shape[0])
    sum_dist_sq, labels, centers = cv2.kmeans(z, no_clusters, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)

    for i, center in enumerate(centers):
        points = z[np.where(labels[:, 0] == i)]
        xmax = int(points[:, 1].max())
        ymax = int(points[:, 0].max())
        xmin = int(points[:, 1].min())
        ymin = int(points[:, 0].min())
        w = xmax - xmin
        h = ymax - ymin
        yield xmin, ymin, w, h


# Part 2: server ----
def server(input: Any, output: Any, session: Any) -> Any:  # noqa: A002, ARG001
    @shiny.reactive.Calc
    def get_frame_raw():
        background_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=900, detectShadows=False, dist2Threshold=input.knn()
        )
        background_subtractor_mog = cv2.createBackgroundSubtractorMOG2(
            history=900, detectShadows=False, varThreshold=input.mog()
        )

        kernel_open = np.ones((input.kernel_open_size(), input.kernel_open_size()), np.uint8)
        kernel_close = np.ones((input.kernel_close_size(), input.kernel_close_size()), np.uint8)

        if input.precalc():
            cap = cv2.VideoCapture(filename)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 0
            while True:
                i += 1
                ret, frame = cap.read()
                if not ret:
                    break
                if not i % 50:
                    print(f"frame {i}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = downscale_image(frame, input.downscale())

                if input.equalization():
                    clip = input.equalization_clip()
                    clahe = cv2.createCLAHE(
                        clipLimit=clip, tileGridSize=(input.equalization_tiles(), input.equalization_tiles())
                    )
                    frame = clahe.apply(frame)

                background_subtractor_knn.apply(frame, learningRate=1.0 / video_length)  # learningRate bugged?
                background_subtractor_mog.apply(frame, learningRate=1.0 / video_length)
            cap.release()

        track_window_knn = None
        track_window_mog = None

        cap = cv2.VideoCapture(filename)
        i = 0
        while i < input.frame_number():
            i += 1
            ret, frame_raw = cap.read()
            if not ret:
                break

            if not i % 50:
                print(f"frame {i}")
            frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
            frame = downscale_image(frame, input.downscale())

            if input.equalization():
                clip = input.equalization_clip()
                clahe = cv2.createCLAHE(
                    clipLimit=clip, tileGridSize=(input.equalization_tiles(), input.equalization_tiles())
                )
                frame = clahe.apply(frame)

            if input.precalc():
                mask_knn = background_subtractor_knn.apply(frame, learningRate=0)  # learningRate bugged?
                mask_mog = background_subtractor_mog.apply(frame, learningRate=0)
            else:
                mask_knn = background_subtractor_knn.apply(frame)
                mask_mog = background_subtractor_mog.apply(frame)

            # denoise see https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
            mask_knn = cv2.morphologyEx(mask_knn, cv2.MORPH_OPEN, kernel_open)
            mask_knn = cv2.morphologyEx(mask_knn, cv2.MORPH_CLOSE, kernel_close)

            mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN, kernel_open)
            mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_CLOSE, kernel_close)

            clusters = tuple(get_clusters(mask_knn, 3))
            cluster_knn = clusters[0] if clusters else None

            clusters = tuple(get_clusters(mask_mog, 3))
            cluster_mog = clusters[0] if clusters else None

            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)

            if not track_window_knn and cluster_knn:
                track_window_knn = cluster_knn
            rect_knn, new_track_window_knn = cv2.CamShift(mask_knn, track_window_knn, term_crit)
            if new_track_window_knn:
                track_window_knn = new_track_window_knn

            if not track_window_mog and cluster_mog:
                track_window_mog = cluster_mog
            if track_window_mog:
                rect_mog, new_track_window_mog = cv2.CamShift(mask_mog, track_window_mog, term_crit)
                if new_track_window_mog:
                    track_window_mog = new_track_window_mog

        frame_boxed_knn = frame.copy()
        frame_boxed_mog = frame.copy()

        if cluster_knn:
            frame_boxed_knn = cv2.rectangle(
                frame_boxed_knn,
                (cluster_knn[0], cluster_knn[1]),
                (cluster_knn[0] + cluster_knn[2], cluster_knn[1] + cluster_knn[3]),
                128,
                1,
            )
        if rect_knn:
            frame_boxed_knn = cv2.polylines(frame_boxed_knn, [np.intp(cv2.boxPoints(rect_knn))], True, 255, 1)

        if cluster_mog:
            frame_boxed_mog = cv2.rectangle(
                frame_boxed_mog,
                (cluster_mog[0], cluster_mog[1]),
                (cluster_mog[0] + cluster_mog[2], cluster_mog[1] + cluster_mog[3]),
                128,
                1,
            )
        if rect_mog:
            frame_boxed_mog = cv2.polylines(frame_boxed_mog, [np.intp(cv2.boxPoints(rect_mog))], True, 255, 1)

        cap.release()
        return (
            frame,
            background_subtractor_knn.getBackgroundImage(),
            mask_knn,
            frame_boxed_knn,
            background_subtractor_mog.getBackgroundImage(),
            mask_mog,
            frame_boxed_mog,
        )

    @output
    @shiny.render.ui
    def result():
        frame, bg_knn, mask_knn, frame_boxed_knn, bg_mog, mask_mog, frame_boxed_mog = get_frame_raw()

        return shiny.ui.page_fluid(
            shiny.ui.tags.div(shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(frame)}", width="49%")),
            shiny.ui.layout_column_wrap(
                (
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(bg_knn)}"),
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(mask_knn)}"),
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(frame_boxed_knn)}"),
                ),
                (
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(bg_mog)}"),
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(mask_mog)}"),
                    shiny.ui.img(src=f"data:image/jpeg;base64, {image_to_b64str(frame_boxed_mog)}"),
                ),
            ),
        )


# Combine into a shiny app.
# hardcode the path to the datafiles for now
www_dir = Path(__file__).parent.parent
# NB: variable must be "app".
app = shiny.App(app_ui, server, static_assets=www_dir)
