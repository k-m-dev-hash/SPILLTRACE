import math


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two geographic coordinates
    in kilometers.
    """

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


# ============================================================
# SAMPLE AIS VESSELS
# ============================================================

def get_sample_vessels():
    """
    Development-only AIS vessel dataset.

    IMPORTANT:
    These are simulated vessels for demonstration/testing.
    They are NOT live AIS observations.
    """

    return [

        {
            "mmsi": "DEV001",
            "vessel_name": "MV Test Vessel",
            "vessel_type": "TANKER",

            # Development study area
            "latitude": 25.6350,
            "longitude": 54.6100,

            "speed_knots": 8.5,
            "heading_degrees": 180
        },

        {
            "mmsi": "DEV002",
            "vessel_name": "MV Chemical Star",
            "vessel_type": "CHEMICAL TANKER",

            "latitude": 25.6200,
            "longitude": 54.5900,

            "speed_knots": 6.0,
            "heading_degrees": 270
        },

        {
            "mmsi": "DEV003",
            "vessel_name": "MV Ocean Carrier",
            "vessel_type": "CARGO",

            "latitude": 25.6450,
            "longitude": 54.6250,

            "speed_knots": 12.0,
            "heading_degrees": 90
        }

    ]


# ============================================================
# CALCULATE VESSEL / CANDIDATE RELATION
# ============================================================

def calculate_vessel_candidate(
    spill_latitude,
    spill_longitude,
    vessel
):
    """
    Calculate distance between a SAR candidate and
    a development AIS vessel.
    """

    distance = haversine_distance(
        spill_latitude,
        spill_longitude,
        vessel["latitude"],
        vessel["longitude"]
    )

    return {
        "mmsi": vessel["mmsi"],
        "vessel_name": vessel["vessel_name"],
        "vessel_type": vessel["vessel_type"],
        "latitude": vessel["latitude"],
        "longitude": vessel["longitude"],
        "speed_knots": vessel["speed_knots"],
        "heading_degrees": vessel["heading_degrees"],
        "distance_to_spill_km": round(
            distance,
            3
        )
    }


# ============================================================
# AIS ATTRIBUTION PRIORITY
# ============================================================

def calculate_attribution_score(
    distance_km,
    vessel_type,
    speed_knots
):
    """
    Calculate a heuristic attribution priority.

    This is NOT proof that the vessel caused the spill.

    It only prioritizes vessels for investigation.
    """

    score = 0

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    if distance_km <= 2:
        score += 50

    elif distance_km <= 5:
        score += 40

    elif distance_km <= 10:
        score += 30

    elif distance_km <= 20:
        score += 20

    else:
        score += 5

    # --------------------------------------------------------
    # VESSEL TYPE
    # --------------------------------------------------------

    vessel_type_upper = (
        vessel_type or ""
    ).upper()

    if vessel_type_upper == "CHEMICAL TANKER":
        score += 30

    elif vessel_type_upper == "TANKER":
        score += 25

    elif vessel_type_upper == "CARGO":
        score += 10

    else:
        score += 5

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    if 2 <= speed_knots <= 15:
        score += 15

    elif speed_knots < 2:
        score += 5

    else:
        score += 8

    # --------------------------------------------------------
    # LIMIT SCORE
    # --------------------------------------------------------

    score = min(
        100,
        max(0, score)
    )

    return {
        "score": score,
        "priority": (
            "HIGH"
            if score >= 70
            else
            "MEDIUM"
            if score >= 40
            else
            "LOW"
        ),
        "note": (
            "Heuristic prioritization only; "
            "not causal proof."
        )
    }