def create_current_data(
    speed_knots,
    direction_degrees,
    source="DEVELOPMENT_DATA"
):
    """
    Create standardized ocean-current information.

    direction_degrees represents the direction
    toward which the current is moving.

    0   = North
    90  = East
    180 = South
    270 = West
    """

    direction = float(direction_degrees)

    return {
        "speed_knots": float(speed_knots),
        "direction_degrees": direction,
        "current_to_direction": direction,
        "source": str(source)
    }


def calculate_direction_difference(
    direction1,
    direction2
):
    """
    Calculate the smallest angular difference
    between two directions.

    Result is between 0 and 180 degrees.
    """

    difference = abs(
        float(direction1) - float(direction2)
    ) % 360

    if difference > 180:
        difference = 360 - difference

    return float(difference)


def calculate_current_alignment(
    spill_orientation,
    current_direction
):
    """
    Compare SAR candidate orientation with
    ocean-current direction.

    SAR orientation represents an axis, so directions
    separated by 180 degrees are treated as the same
    orientation.

    This is a heuristic indicator, not a physical
    ocean-drift model.
    """

    spill_orientation = (
        float(spill_orientation) % 180
    )

    current_direction = (
        float(current_direction) % 180
    )

    difference = abs(
        spill_orientation -
        current_direction
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
        "direction_difference":
            round(difference, 2),

        "alignment_score":
            score,

        "classification":
            classification
    }


def analyze_current_for_candidate(
    candidate,
    current_data
):
    """
    Analyze ocean-current conditions
    for a SAR candidate.
    """

    if not current_data:

        return {
            "available": False,
            "message":
                "Current data not available"
        }

    speed = float(
        current_data["speed_knots"]
    )

    direction = float(
        current_data["current_to_direction"]
    )

    alignment = calculate_current_alignment(
        candidate["orientation"],
        direction
    )

    if speed < 0.5:

        speed_classification = "WEAK CURRENT"

    elif speed <= 2:

        speed_classification = "MODERATE CURRENT"

    else:

        speed_classification = "STRONG CURRENT"

    return {

        "available":
            True,

        "speed_knots":
            speed,

        "current_to_direction":
            direction,

        "source":
            current_data.get(
                "source",
                "UNKNOWN"
            ),

        "speed_classification":
            speed_classification,

        "alignment":
            alignment
    }