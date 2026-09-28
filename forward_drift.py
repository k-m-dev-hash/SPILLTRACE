"""
SPILLTRACE - Forward Drift Validation
=====================================

Prototype forward drift simulation used to compare a candidate
vessel trajectory against the observed SAR spill location.

IMPORTANT:
This is a prototype compatibility model, NOT a physical ocean
circulation model and NOT proof of vessel causality.
"""

import math


# ============================================================
# GEOSPATIAL UTILITIES
# ============================================================

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance between two coordinates.
    """

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return EARTH_RADIUS_KM * c


def destination_point(
    latitude,
    longitude,
    bearing_degrees,
    distance_km,
):
    """
    Move a geographic point along a bearing by a distance.
    """

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    bearing = math.radians(bearing_degrees)

    angular_distance = (
        distance_km / EARTH_RADIUS_KM
    )

    lat2 = math.asin(
        math.sin(lat1)
        * math.cos(angular_distance)
        +
        math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),
        math.cos(angular_distance)
        -
        math.sin(lat1)
        * math.sin(lat2),
    )

    return {
        "latitude": math.degrees(lat2),
        "longitude": math.degrees(lon2),
    }


# ============================================================
# DIRECTION UTILITIES
# ============================================================

def direction_difference(
    direction_a,
    direction_b,
):
    """
    Smallest angular difference between two bearings.
    """

    difference = abs(
        (direction_a - direction_b + 180)
        % 360
        - 180
    )

    return difference


def vector_components(
    speed_knots,
    direction_degrees,
):
    """
    Convert speed + bearing into north/east components.
    """

    direction = math.radians(
        direction_degrees
    )

    north = (
        speed_knots
        * math.cos(direction)
    )

    east = (
        speed_knots
        * math.sin(direction)
    )

    return north, east


def combined_drift_vector(
    wind_speed_knots,
    wind_from_direction,
    current_speed_knots,
    current_to_direction,
    wind_factor=0.03,
):
    """
    Combine wind-driven and ocean-current movement.

    Wind direction is supplied as FROM direction and converted
    to the corresponding TO direction.
    """

    wind_to_direction = (
        wind_from_direction + 180
    ) % 360

    wind_north, wind_east = vector_components(
        wind_speed_knots * wind_factor,
        wind_to_direction,
    )

    current_north, current_east = vector_components(
        current_speed_knots,
        current_to_direction,
    )

    north = wind_north + current_north
    east = wind_east + current_east

    magnitude = math.sqrt(
        north ** 2 + east ** 2
    )

    bearing = (
        math.degrees(
            math.atan2(east, north)
        )
        % 360
    )

    return {
        "north_knots": north,
        "east_knots": east,
        "speed_knots": magnitude,
        "bearing_degrees": bearing,
        "wind_to_direction": wind_to_direction,
    }


# ============================================================
# FORWARD DRIFT SIMULATION
# ============================================================

def simulate_forward_drift(
    vessel_lat,
    vessel_lon,
    wind_speed_knots=8.0,
    wind_from_direction=270.0,
    current_speed_knots=1.2,
    current_to_direction=90.0,
    hours=6,
    timestep_hours=1,
):
    """
    Simulate forward movement from a candidate vessel.

    Returns a trajectory from T+0 to T+N hours.
    """

    drift = combined_drift_vector(
        wind_speed_knots=wind_speed_knots,
        wind_from_direction=wind_from_direction,
        current_speed_knots=current_speed_knots,
        current_to_direction=current_to_direction,
    )

    trajectory = []

    current_position = {
        "latitude": float(vessel_lat),
        "longitude": float(vessel_lon),
    }

    total_steps = int(
        hours / timestep_hours
    )

    for step in range(total_steps + 1):

        elapsed_hours = (
            step * timestep_hours
        )

        trajectory.append(
            {
                "time_hours": elapsed_hours,
                "latitude": current_position[
                    "latitude"
                ],
                "longitude": current_position[
                    "longitude"
                ],
            }
        )

        if step == total_steps:
            break

        distance_km = (
            drift["speed_knots"]
            * 1.852
            * timestep_hours
        )

        current_position = destination_point(
            latitude=current_position[
                "latitude"
            ],
            longitude=current_position[
                "longitude"
            ],
            bearing_degrees=drift[
                "bearing_degrees"
            ],
            distance_km=distance_km,
        )

    return {
        "trajectory": trajectory,
        "predicted_position": trajectory[-1],
        "parameters": {
            "hours": hours,
            "timestep_hours": timestep_hours,
            "wind_speed_knots": wind_speed_knots,
            "wind_from_direction": wind_from_direction,
            "current_speed_knots": current_speed_knots,
            "current_to_direction": current_to_direction,
            "wind_factor": 0.03,
        },
        "drift": drift,
        "notes": [
            "Prototype forward drift simulation",
            "Uses synthetic wind/current conditions",
            "Not a validated ocean circulation model",
            "Does not establish vessel causality",
        ],
    }


# ============================================================
# VALIDATION AGAINST OBSERVED SAR SPILL
# ============================================================

def validate_forward_drift(
    vessel_lat,
    vessel_lon,
    spill_lat,
    spill_lon,
    wind_speed_knots=8.0,
    wind_from_direction=270.0,
    current_speed_knots=1.2,
    current_to_direction=90.0,
    hours=6,
):
    """
    Simulate forward drift from a candidate vessel and compare
    the predicted position against the observed SAR spill.
    """

    simulation = simulate_forward_drift(
        vessel_lat=vessel_lat,
        vessel_lon=vessel_lon,
        wind_speed_knots=wind_speed_knots,
        wind_from_direction=wind_from_direction,
        current_speed_knots=current_speed_knots,
        current_to_direction=current_to_direction,
        hours=hours,
        timestep_hours=1,
    )

    predicted = simulation[
        "predicted_position"
    ]

    distance_km = haversine_distance_km(
        predicted["latitude"],
        predicted["longitude"],
        spill_lat,
        spill_lon,
    )

    # --------------------------------------------------------
    # Compatibility score
    # --------------------------------------------------------

    # 0 km = 100
    # 20+ km = 0
    distance_score = max(
        0.0,
        100.0
        * (
            1.0
            - distance_km / 20.0
        ),
    )

    if distance_km <= 3:
        validation_level = "HIGH"

    elif distance_km <= 10:
        validation_level = "MEDIUM"

    else:
        validation_level = "LOW"

    return {
        "validated": True,

        "candidate_vessel_position": {
            "latitude": float(vessel_lat),
            "longitude": float(vessel_lon),
        },

        "observed_spill_position": {
            "latitude": float(spill_lat),
            "longitude": float(spill_lon),
        },

        "predicted_spill_position": predicted,

        "distance_to_observed_spill_km": round(
            distance_km,
            3,
        ),

        "forward_drift_score": round(
            distance_score,
            2,
        ),

        "validation_level": validation_level,

        "simulation": simulation,
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SPILLTRACE FORWARD DRIFT VALIDATION TEST")
    print("=" * 70)

    # Example candidate vessel
    vessel = {
        "latitude": 25.43534,
        "longitude": 54.49016,
    }

    # Example observed SAR spill
    spill = {
        "latitude": 25.435356,
        "longitude": 54.649509,
    }

    result = validate_forward_drift(
        vessel_lat=vessel["latitude"],
        vessel_lon=vessel["longitude"],
        spill_lat=spill["latitude"],
        spill_lon=spill["longitude"],
        wind_speed_knots=8.0,
        wind_from_direction=270.0,
        current_speed_knots=1.2,
        current_to_direction=90.0,
        hours=6,
    )

    print()
    print(
        "Candidate vessel:",
        vessel,
    )

    print(
        "Observed spill:",
        spill,
    )

    print()
    print(
        "Predicted spill:",
        result[
            "predicted_spill_position"
        ],
    )

    print(
        "Distance to observed spill:",
        result[
            "distance_to_observed_spill_km"
        ],
        "km",
    )

    print(
        "Forward drift score:",
        result[
            "forward_drift_score"
        ],
    )

    print(
        "Validation:",
        result[
            "validation_level"
        ],
    )

    print()
    print("Trajectory:")

    for point in result[
        "simulation"
    ]["trajectory"]:

        print(
            f"  T+{point['time_hours']}h"
            f" -> "
            f"{point['latitude']:.6f}, "
            f"{point['longitude']:.6f}"
        )

    print("=" * 70)