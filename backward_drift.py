# ============================================================
# SPILLTRACE - Backward Drift / Source Reconstruction
# SIH26143
#
# Prototype environmental backtracking model.
#
# IMPORTANT:
# This is NOT a validated ocean circulation model.
# Wind/current values are currently prototype inputs.
# ============================================================

from math import radians, sin, cos, sqrt, atan2, degrees


# ============================================================
# CONSTANTS
# ============================================================

EARTH_RADIUS_KM = 6371.0
KNOT_TO_KM_H = 1.852


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        +
        cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return EARTH_RADIUS_KM * c


# ============================================================
# DESTINATION POINT
# ============================================================

def destination_point(
    latitude,
    longitude,
    distance_km,
    bearing_degrees,
):
    """
    Calculate a destination coordinate given:

        start coordinate
        distance
        bearing
    """

    lat1 = radians(float(latitude))
    lon1 = radians(float(longitude))

    bearing = radians(
        float(bearing_degrees)
    )

    angular_distance = (
        distance_km
        / EARTH_RADIUS_KM
    )

    lat2 = atan2(
        sin(lat1)
        * cos(angular_distance)
        +
        cos(lat1)
        * sin(angular_distance)
        * cos(bearing),

        sqrt(
            (
                1
                -
                (
                    sin(lat1)
                    * cos(angular_distance)
                    +
                    cos(lat1)
                    * sin(angular_distance)
                    * cos(bearing)
                )
                ** 2
            )
        ),
    )

    lon2 = (
        lon1
        +
        atan2(
            sin(bearing)
            * sin(angular_distance)
            * cos(lat1),

            cos(angular_distance)
            -
            sin(lat1)
            * sin(lat2),
        )
    )

    return {
        "latitude": degrees(lat2),
        "longitude": degrees(lon2),
    }


# ============================================================
# VECTOR COMPONENTS
# ============================================================

def vector_components(
    speed_knots,
    direction_degrees,
):
    """
    Convert speed + bearing into east/north components.

    direction_degrees is the direction TOWARD which the
    movement occurs.
    """

    speed_kmh = (
        float(speed_knots)
        * KNOT_TO_KM_H
    )

    angle = radians(
        float(direction_degrees)
    )

    east = (
        speed_kmh
        * sin(angle)
    )

    north = (
        speed_kmh
        * cos(angle)
    )

    return east, north


# ============================================================
# COMBINE ENVIRONMENTAL VECTORS
# ============================================================

def combined_drift_vector(
    wind_speed_knots,
    wind_direction_to,
    current_speed_knots,
    current_direction_to,
    wind_factor=0.03,
):
    """
    Combine ocean-current movement and a simplified wind
    contribution.

    wind_factor = fraction of wind speed applied to slick.

    This is a prototype approximation.
    """

    # --------------------------------------------------------
    # Ocean current
    # --------------------------------------------------------

    current_east, current_north = (
        vector_components(
            current_speed_knots,
            current_direction_to,
        )
    )

    # --------------------------------------------------------
    # Simplified wind-induced surface drift
    # --------------------------------------------------------

    effective_wind_speed = (
        float(wind_speed_knots)
        * wind_factor
    )

    wind_east, wind_north = (
        vector_components(
            effective_wind_speed,
            wind_direction_to,
        )
    )

    total_east = (
        current_east
        + wind_east
    )

    total_north = (
        current_north
        + wind_north
    )

    return {
        "east_kmh": total_east,
        "north_kmh": total_north,
    }


# ============================================================
# VECTOR MAGNITUDE
# ============================================================

def vector_magnitude(
    east,
    north,
):
    return sqrt(
        east ** 2
        +
        north ** 2
    )


# ============================================================
# VECTOR BEARING
# ============================================================

def vector_bearing(
    east,
    north,
):
    """
    Convert east/north vector to bearing.
    """

    bearing = degrees(
        atan2(
            east,
            north,
        )
    )

    return (
        bearing + 360
    ) % 360


# ============================================================
# BACKWARD DRIFT
# ============================================================

def calculate_backward_drift(
    spill_lat,
    spill_lon,
    wind_speed_knots=8.0,
    wind_direction_from=270.0,
    current_speed_knots=1.2,
    current_direction_to=90.0,
    hours=6,
    time_step_hours=1,
):
    """
    Reconstruct a possible origin path backward from the
    detected spill.

    Prototype assumptions:

        Wind:
            8 knots FROM 270°
            therefore movement contribution is toward 90°

        Ocean current:
            1.2 knots toward 90°

    The backward trajectory moves opposite to the estimated
    forward drift vector.
    """

    # --------------------------------------------------------
    # Convert wind FROM direction to wind TO direction.
    #
    # Example:
    # FROM 270° -> TO 90°
    # --------------------------------------------------------

    wind_direction_to = (
        float(wind_direction_from)
        + 180
    ) % 360

    # --------------------------------------------------------
    # Calculate combined forward drift vector.
    # --------------------------------------------------------

    vector = combined_drift_vector(

        wind_speed_knots=
            wind_speed_knots,

        wind_direction_to=
            wind_direction_to,

        current_speed_knots=
            current_speed_knots,

        current_direction_to=
            current_direction_to,

    )

    forward_east = vector[
        "east_kmh"
    ]

    forward_north = vector[
        "north_kmh"
    ]

    forward_speed = vector_magnitude(
        forward_east,
        forward_north,
    )

    forward_direction = vector_bearing(
        forward_east,
        forward_north,
    )

    # --------------------------------------------------------
    # Backward direction = opposite direction.
    # --------------------------------------------------------

    backward_direction = (
        forward_direction
        + 180
    ) % 360

    # --------------------------------------------------------
    # Generate trajectory.
    # --------------------------------------------------------

    trajectory = [

        {
            "hour": 0,
            "latitude": float(
                spill_lat
            ),
            "longitude": float(
                spill_lon
            ),
            "distance_from_spill_km": 0.0,
        }

    ]

    current_lat = float(
        spill_lat
    )

    current_lon = float(
        spill_lon
    )

    elapsed = 0

    while elapsed < hours:

        step = min(
            time_step_hours,
            hours - elapsed,
        )

        distance = (
            forward_speed
            * step
        )

        point = destination_point(

            current_lat,

            current_lon,

            distance,

            backward_direction,

        )

        current_lat = point[
            "latitude"
        ]

        current_lon = point[
            "longitude"
        ]

        elapsed += step

        trajectory.append({

            "hour": elapsed,

            "latitude": current_lat,

            "longitude": current_lon,

            "distance_from_spill_km":
                round(
                    haversine_distance_km(
                        spill_lat,
                        spill_lon,
                        current_lat,
                        current_lon,
                    ),
                    3,
                ),

        })

    # --------------------------------------------------------
    # Final reconstructed source zone.
    # --------------------------------------------------------

    source = trajectory[-1]

    # --------------------------------------------------------
    # Uncertainty estimate.
    #
    # Prototype only.
    # --------------------------------------------------------

    uncertainty_radius = max(
        2.0,
        forward_speed
        * hours
        * 0.20,
    )

    return {

        "status":
            "PROTOTYPE_ESTIMATE",

        "model":
            "SPILLTRACE Backward Drift Prototype",

        "spill_location": {

            "latitude":
                float(spill_lat),

            "longitude":
                float(spill_lon),

        },

        "estimated_source": {

            "latitude":
                source["latitude"],

            "longitude":
                source["longitude"],

        },

        "trajectory":
            trajectory,

        "parameters": {

            "wind_speed_knots":
                float(
                    wind_speed_knots
                ),

            "wind_direction_from":
                float(
                    wind_direction_from
                ),

            "wind_direction_to":
                wind_direction_to,

            "current_speed_knots":
                float(
                    current_speed_knots
                ),

            "current_direction_to":
                float(
                    current_direction_to
                ),

            "wind_drift_factor":
                0.03,

            "hours":
                hours,

            "time_step_hours":
                time_step_hours,

        },

        "drift_vector": {

            "forward_speed_kmh":
                round(
                    forward_speed,
                    3,
                ),

            "forward_direction_to":
                round(
                    forward_direction,
                    2,
                ),

            "backward_direction_to":
                round(
                    backward_direction,
                    2,
                ),

        },

        "uncertainty": {

            "radius_km":
                round(
                    uncertainty_radius,
                    2,
                ),

            "type":
                "Prototype heuristic",

        },

        "notes": [

            (
                "Backward trajectory is a "
                "prototype environmental "
                "approximation."
            ),

            (
                "Wind and current values are "
                "prototype inputs and are not "
                "live ERA5/Copernicus data."
            ),

            (
                "The estimated source zone does "
                "not establish spill causality."
            ),

        ],

    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    result = calculate_backward_drift(

        spill_lat=3.348711,

        spill_lon=104.195174,

        wind_speed_knots=8,

        wind_direction_from=270,

        current_speed_knots=1.2,

        current_direction_to=90,

        hours=6,

        time_step_hours=1,

    )

    print()

    print("=" * 70)

    print(
        "SPILLTRACE BACKWARD DRIFT TEST"
    )

    print("=" * 70)

    print(
        "Spill:",
        result["spill_location"],
    )

    print(
        "Estimated source:",
        result["estimated_source"],
    )

    print(
        "Forward drift:",
        result[
            "drift_vector"
        ]["forward_speed_kmh"],
        "km/h",
    )

    print(
        "Forward direction:",
        result[
            "drift_vector"
        ]["forward_direction_to"],
        "degrees",
    )

    print(
        "Backward direction:",
        result[
            "drift_vector"
        ]["backward_direction_to"],
        "degrees",
    )

    print(
        "Uncertainty radius:",
        result[
            "uncertainty"
        ]["radius_km"],
        "km",
    )

    print()

    print("Trajectory:")

    for point in result[
        "trajectory"
    ]:

        print(

            f"  T-{point['hour']}h -> "

            f"{point['latitude']:.6f}, "

            f"{point['longitude']:.6f} "

            f"({point['distance_from_spill_km']:.2f} km)"

        )

    print("=" * 70)