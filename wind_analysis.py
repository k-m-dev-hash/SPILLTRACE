def create_wind_data(
    speed_knots,
    direction_degrees,
    source="DEVELOPMENT_DATA"
):
    """
    Create standardized meteorological wind information.

    direction_degrees represents the direction the wind
    is coming FROM.

    0   = North
    90  = East
    180 = South
    270 = West
    """

    wind_from = float(direction_degrees)

    # Convert meteorological "FROM" direction
    # into movement "TO" direction.
    wind_to = (wind_from + 180) % 360

    return {
        "speed_knots": float(speed_knots),
        "direction_degrees": wind_from,
        "wind_from_direction": wind_from,
        "wind_to_direction": wind_to,
        "source": str(source)
    }


def calculate_direction_difference(direction1, direction2):
    """
    Calculate the smallest angular difference between
    two directions.

    Result is between 0 and 180 degrees.
    """

    difference = abs(
        float(direction1) - float(direction2)
    ) % 360

    if difference > 180:
        difference = 360 - difference

    return float(difference)


def calculate_wind_alignment(
    spill_orientation,
    wind_direction
):
    """
    Compare SAR candidate orientation with wind direction.

    SAR orientation represents an axis, so directions separated
    by 180 degrees are treated as the same orientation.

    This is a heuristic indicator, not a physical oil-drift model.
    """

    spill_orientation = float(spill_orientation) % 180
    wind_direction = float(wind_direction) % 180

    difference = abs(
        spill_orientation - wind_direction
    )

    if difference > 90:
        difference = 180 - difference

    if difference <= 30:
        score = 100
        classification = "STRONG ALIGNMENT"

    elif difference <= 60:
        score = 75
        classification = "MODERATE ALIGNMENT"

    elif difference <= 90:
        score = 50
        classification = "WEAK ALIGNMENT"

    else:
        score = 25
        classification = "LOW ALIGNMENT"

    return {
        "direction_difference": round(
            difference,
            2
        ),
        "alignment_score": score,
        "classification": classification
    }

def analyze_wind_for_candidate(
    candidate,
    wind_data
):
    """
    Analyze wind conditions for a SAR candidate.
    """

    if not wind_data:

        return {
            "available": False,
            "message": "Wind data not available"
        }

    speed = float(
        wind_data["speed_knots"]
    )

    wind_from = float(
        wind_data["wind_from_direction"]
    )

    wind_to = float(
        wind_data["wind_to_direction"]
    )

    # Use wind-to direction for movement alignment.
    alignment = calculate_wind_alignment(
        candidate["orientation"],
        wind_to
    )

    if speed < 2:

        speed_classification = "LOW WIND"

    elif speed <= 10:

        speed_classification = "MODERATE WIND"

    else:

        speed_classification = "STRONG WIND"

    return {

        "available":
            True,

        "speed_knots":
            speed,

        "wind_from_direction":
            wind_from,

        "wind_to_direction":
            wind_to,

        "source":
            wind_data.get(
                "source",
                "UNKNOWN"
            ),

        "speed_classification":
            speed_classification,

        "alignment":
            alignment
    }