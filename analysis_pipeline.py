from modules.spill_detection import detect_dark_regions
from modules.lookalike import validate_candidate
from modules.wind_analysis import analyze_wind_for_candidate
from modules.current_analysis import analyze_current_for_candidate


def analyze_sentinel_image(
    image_path,
    wind_data=None,
    current_data=None,
    wind_speed=None
):
    detection = detect_dark_regions(image_path)

    # --------------------------------------------------------
    # Backward compatibility for wind_speed
    # --------------------------------------------------------

    if wind_data is None and wind_speed is not None:

        wind_data = {
            "speed_knots": float(wind_speed),
            "direction_degrees": 0.0,
            "wind_from_direction": 0.0,
            "wind_to_direction": 180.0,
            "source": "DEVELOPMENT_DATA"
        }

    candidates = []

    for candidate in detection["candidates"]:

        # ----------------------------------------------------
        # SAR validation
        # ----------------------------------------------------

        validation = validate_candidate(
            candidate,
            detection["image_statistics"],
            wind_speed=(
                wind_data["speed_knots"]
                if wind_data
                else None
            )
        )

        # ----------------------------------------------------
        # Wind analysis
        # ----------------------------------------------------

        wind_analysis = analyze_wind_for_candidate(
            candidate,
            wind_data
        )

        # ----------------------------------------------------
        # Ocean-current analysis
        # ----------------------------------------------------

        current_analysis = (
            analyze_current_for_candidate(
                candidate,
                current_data
            )
        )

        candidates.append({

            "candidate":
                candidate,

            "validation":
                validation,

            "wind_analysis":
                wind_analysis,

            "current_analysis":
                current_analysis
        })

    return {

        "image_width":
            detection["image_width"],

        "image_height":
            detection["image_height"],

        "image_statistics":
            detection["image_statistics"],

        "candidate_count":
            len(candidates),

        "candidates":
            candidates,

        "wind": {

            "available":
                bool(wind_data),

            "data":
                wind_data
        },

        "current": {

            "available":
                bool(current_data),

            "data":
                current_data
        }
    }