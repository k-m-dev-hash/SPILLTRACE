# ============================================================
# SPILLTRACE - AIS Correlation Prototype
# SIH26143
#
# Satellite-AIS Intelligence for Oil Spill Source Attribution
#
# NOTE:
# This module currently uses SYNTHETIC AIS data for prototype
# demonstration. It is NOT live AIS data and does NOT prove
# vessel causality.
# ============================================================

from math import radians, sin, cos, sqrt, atan2, degrees
from datetime import datetime, timezone


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate great-circle distance between two coordinates.

    Returns:
        Distance in kilometres.
    """

    R = 6371.0

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

    return R * c


# ============================================================
# BEARING
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate initial bearing from point 1 to point 2.

    Returns:
        Bearing in degrees [0, 360).
    """

    lat1 = radians(float(lat1))
    lat2 = radians(float(lat2))

    dlon = radians(
        float(lon2) - float(lon1)
    )

    x = (
        sin(dlon)
        * cos(lat2)
    )

    y = (
        cos(lat1)
        * sin(lat2)
        -
        sin(lat1)
        * cos(lat2)
        * cos(dlon)
    )

    bearing = degrees(
        atan2(x, y)
    )

    return (
        bearing + 360
    ) % 360


# ============================================================
# ANGLE DIFFERENCE
# ============================================================

def angle_difference(
    angle1,
    angle2,
):
    """
    Calculate smallest angular difference.

    Example:
        350 vs 10 -> 20 degrees
    """

    difference = abs(
        float(angle1)
        -
        float(angle2)
    )

    return min(
        difference,
        360 - difference,
    )


# ============================================================
# SYNTHETIC AIS DATA
# ============================================================

def generate_demo_ais(
    spill_lat,
    spill_lon,
):
    """
    Generate synthetic AIS contacts around the detected spill.

    This is ONLY prototype/demo data.

    Later this function can be replaced with real AIS
    retrieval from MarineCadastre / AccessAIS or another
    approved AIS source.
    """

    return [

        {
            "mmsi": "419001001",
            "vessel_name": "MV Ocean Star",
            "vessel_type": "Cargo",
            "latitude": spill_lat + 0.025,
            "longitude": spill_lon - 0.015,
            "speed_knots": 14.8,
            "heading": 135,
            "course": 135,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "demo": True,
        },

        {
            "mmsi": "419001002",
            "vessel_name": "MV Blue Horizon",
            "vessel_type": "Tanker",
            "latitude": spill_lat - 0.045,
            "longitude": spill_lon + 0.020,
            "speed_knots": 11.6,
            "heading": 315,
            "course": 315,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "demo": True,
        },

        {
            "mmsi": "419001003",
            "vessel_name": "MV Coastal Star",
            "vessel_type": "Container",
            "latitude": spill_lat + 0.090,
            "longitude": spill_lon + 0.060,
            "speed_knots": 18.2,
            "heading": 250,
            "course": 250,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "demo": True,
        },

        {
            "mmsi": "419001004",
            "vessel_name": "MT Ocean Trader",
            "vessel_type": "Oil Tanker",
            "latitude": spill_lat - 0.110,
            "longitude": spill_lon - 0.070,
            "speed_knots": 9.4,
            "heading": 45,
            "course": 45,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "demo": True,
        },

        {
            "mmsi": "419001005",
            "vessel_name": "MV Sea Falcon",
            "vessel_type": "Cargo",
            "latitude": spill_lat + 0.160,
            "longitude": spill_lon - 0.130,
            "speed_knots": 20.1,
            "heading": 180,
            "course": 180,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "demo": True,
        },

    ]


# ============================================================
# DISTANCE SCORE
# ============================================================

def calculate_distance_score(
    distance_km,
):
    """
    Prototype proximity score.

    Closer vessels receive higher scores.
    """

    distance_km = float(
        distance_km
    )

    if distance_km <= 2:
        return 100.0

    if distance_km <= 5:
        return 90.0

    if distance_km <= 10:
        return 75.0

    if distance_km <= 20:
        return 55.0

    if distance_km <= 40:
        return 30.0

    return 10.0


# ============================================================
# HEADING SCORE
# ============================================================

def calculate_heading_score(
    vessel_heading,
    bearing_to_spill,
):
    """
    Compare vessel heading with direction toward the spill.

    This is a simple prototype feature, not a physical
    trajectory model.
    """

    difference = angle_difference(
        vessel_heading,
        bearing_to_spill,
    )

    if difference <= 15:
        return 100.0

    if difference <= 30:
        return 90.0

    if difference <= 60:
        return 70.0

    if difference <= 90:
        return 50.0

    if difference <= 135:
        return 30.0

    return 10.0


# ============================================================
# SPEED SCORE
# ============================================================

def calculate_speed_score(
    speed_knots,
):
    """
    Prototype vessel-motion plausibility score.

    This is NOT a causal oil-spill model.
    """

    speed = float(
        speed_knots
    )

    if 5 <= speed <= 20:
        return 100.0

    if 2 <= speed < 5:
        return 70.0

    if 20 < speed <= 30:
        return 70.0

    if speed < 2:
        return 45.0

    return 30.0


# ============================================================
# CORRELATE ONE VESSEL
# ============================================================

def correlate_vessel(
    vessel,
    spill_lat,
    spill_lon,
):
    """
    Calculate AIS correlation features for one vessel.
    """

    vessel_lat = float(
        vessel["latitude"]
    )

    vessel_lon = float(
        vessel["longitude"]
    )

    vessel_heading = float(
        vessel.get(
            "heading",
            vessel.get(
                "course",
                0,
            ),
        )
    )

    vessel_speed = float(
        vessel.get(
            "speed_knots",
            0,
        )
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance_km = haversine_distance_km(
        spill_lat,
        spill_lon,
        vessel_lat,
        vessel_lon,
    )

    # --------------------------------------------------------
    # Direction from vessel to spill
    # --------------------------------------------------------

    bearing_to_spill = calculate_bearing(
        vessel_lat,
        vessel_lon,
        spill_lat,
        spill_lon,
    )

    # --------------------------------------------------------
    # Individual evidence scores
    # --------------------------------------------------------

    distance_score = calculate_distance_score(
        distance_km
    )

    heading_score = calculate_heading_score(
        vessel_heading,
        bearing_to_spill,
    )

    speed_score = calculate_speed_score(
        vessel_speed
    )

    # --------------------------------------------------------
    # Evidence fusion
    #
    # Prototype weighting:
    #
    # Distance = 50%
    # Heading  = 30%
    # Speed    = 20%
    # --------------------------------------------------------

    correlation_score = (
        distance_score * 0.50
        +
        heading_score * 0.30
        +
        speed_score * 0.20
    )

    correlation_score = round(
        correlation_score,
        1,
    )

    # --------------------------------------------------------
    # Evidence level
    # --------------------------------------------------------

    if correlation_score >= 80:
        evidence_level = "HIGH"

    elif correlation_score >= 60:
        evidence_level = "MEDIUM"

    else:
        evidence_level = "LOW"

    # --------------------------------------------------------
    # Evidence explanation
    # --------------------------------------------------------

    evidence = [

        f"Distance: {distance_km:.2f} km",

        (
            f"Heading alignment: "
            f"{heading_score:.0f}/100"
        ),

        (
            f"Speed plausibility: "
            f"{speed_score:.0f}/100"
        ),

    ]

    return {

        "mmsi": vessel["mmsi"],

        "vessel_name": vessel[
            "vessel_name"
        ],

        "vessel_type": vessel[
            "vessel_type"
        ],

        "latitude": vessel_lat,

        "longitude": vessel_lon,

        "speed_knots": vessel_speed,

        "heading": vessel_heading,

        "timestamp": vessel.get(
            "timestamp"
        ),

        "distance_km": round(
            distance_km,
            2,
        ),

        "bearing_to_spill": round(
            bearing_to_spill,
            1,
        ),

        "distance_score": round(
            distance_score,
            1,
        ),

        "heading_score": round(
            heading_score,
            1,
        ),

        "speed_score": round(
            speed_score,
            1,
        ),

        "correlation_score": correlation_score,

        "evidence_level": evidence_level,

        "demo_data": vessel.get(
            "demo",
            True,
        ),

        "evidence": evidence,

    }


# ============================================================
# CORRELATE AND RANK AIS VESSELS
# ============================================================

def correlate_ais(
    spill_lat,
    spill_lon,
):
    """
    Generate AIS contacts and rank them against the spill.

    Returns:
        List of ranked vessels.
    """

    vessels = generate_demo_ais(
        spill_lat,
        spill_lon,
    )

    correlated = []

    for vessel in vessels:

        result = correlate_vessel(
            vessel=vessel,
            spill_lat=spill_lat,
            spill_lon=spill_lon,
        )

        correlated.append(
            result
        )

    # --------------------------------------------------------
    # Sort highest correlation first
    # --------------------------------------------------------

    correlated.sort(
        key=lambda vessel:
        vessel["correlation_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Assign ranking
    # --------------------------------------------------------

    for index, vessel in enumerate(
        correlated,
        start=1,
    ):

        vessel["rank"] = index

    return correlated


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    test_lat = 3.348711
    test_lon = 104.195174

    results = correlate_ais(
        test_lat,
        test_lon,
    )

    print()
    print("=" * 70)
    print("SPILLTRACE AIS CORRELATION TEST")
    print("=" * 70)

    for vessel in results:

        print(
            f"#{vessel['rank']} "
            f"{vessel['vessel_name']} "
            f"| {vessel['distance_km']} km "
            f"| score {vessel['correlation_score']} "
            f"| {vessel['evidence_level']}"
        )

    print("=" * 70)