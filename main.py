# ============================================================
# SPILLTRACE
# Satellite-AIS Intelligence for Oil Spill Source Attribution
# ============================================================

import os
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio

from fastapi import (
    FastAPI,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ============================================================
# SPILLTRACE MODULES
# ============================================================

from modules.unet_detector import (
    detect_spill_unet,
    save_prediction_mask,
)

from modules.sar_classifier import (
    classify_sar_image,
)

from modules.ais_correlation import (
    correlate_ais,
)

from modules.backward_drift import (
    calculate_backward_drift,
)

from modules.forward_drift import (
    validate_forward_drift,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SPILLTRACE",
    description=(
        "Satellite-AIS Intelligence for "
        "Oil Spill Source Attribution"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BACKEND_ROOT = Path(
    __file__
).resolve().parent

PREDICTION_DIR = (
    BACKEND_ROOT / "prediction_masks"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/prediction-mask",
    StaticFiles(
        directory=str(PREDICTION_DIR)
    ),
    name="prediction-mask",
)


# ============================================================
# PROTOTYPE ENVIRONMENT
# ============================================================

WIND_SPEED_KNOTS = 8.0
WIND_FROM_DIRECTION = 270.0

CURRENT_SPEED_KNOTS = 1.2
CURRENT_TO_DIRECTION = 90.0

DRIFT_HOURS = 6
DRIFT_TIMESTEP_HOURS = 1


# ============================================================
# SYSTEM FLAGS
# ============================================================

AIS_AVAILABLE = True
BACKWARD_DRIFT_AVAILABLE = True
FORWARD_DRIFT_AVAILABLE = True


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


def generate_spill_id():
    return (
        f"SP-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )


def safe_float(value, default=None):

    try:
        if value is None:
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(value, default=0):

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# GEOSPATIAL UTILITIES
# ============================================================

def pixel_to_latlon(
    image_path,
    x,
    y,
):
    """
    Convert pixel coordinates to geographic
    latitude/longitude using the GeoTIFF transform.
    """

    with rasterio.open(
        image_path
    ) as src:

        lon, lat = rasterio.transform.xy(
            src.transform,
            y,
            x,
            offset="center",
        )

    return {
        "latitude": float(lat),
        "longitude": float(lon),
    }


# ============================================================
# MULTIPLE REGION EXTRACTION
# ============================================================

def extract_spill_regions(
    mask,
    image_path,
    min_area_pixels=500,
):
    """
    Extract connected spill regions from the
    U-Net binary mask.

    This allows the dashboard to display
    multiple candidate regions rather than
    only the largest contour.
    """

    if mask is None:
        return []

    binary_mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    if not np.any(binary_mask):
        return []

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    regions = []

    for label_id in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area_pixels:
            continue

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        centroid_x = float(
            centroids[
                label_id,
                0,
            ]
        )

        centroid_y = float(
            centroids[
                label_id,
                1,
            ]
        )

        location = pixel_to_latlon(
            image_path,
            centroid_x,
            centroid_y,
        )

        contour_mask = (
            labels == label_id
        ).astype(
            np.uint8
        )

        contours, _ = cv2.findContours(
            contour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        largest_contour_area = 0.0

        if contours:
            largest_contour_area = float(
                max(
                    cv2.contourArea(
                        contour
                    )
                    for contour in contours
                )
            )

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        perimeter = 0.0

        if contours:

            perimeter = float(
                cv2.arcLength(
                    max(
                        contours,
                        key=cv2.contourArea,
                    ),
                    True,
                )
            )

        compactness = 0.0

        if perimeter > 0:

            compactness = (
                4.0
                * np.pi
                * largest_contour_area
                / (
                    perimeter
                    * perimeter
                )
            )

        regions.append(
            {
                "region_id": int(
                    label_id
                ),

                "area_pixels": area,

                "largest_contour_area_pixels": (
                    largest_contour_area
                ),

                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },

                "centroid": {
                    "x": centroid_x,
                    "y": centroid_y,
                },

                "aspect_ratio": float(
                    aspect_ratio
                ),

                "compactness": float(
                    compactness
                ),

                "contour_count": len(
                    contours
                ),

                "location": location,
            }
        )

    # Largest region first
    regions.sort(
        key=lambda item: item[
            "area_pixels"
        ],
        reverse=True,
    )

    return regions


# ============================================================
# CLASSIFICATION NORMALIZATION
# ============================================================

def normalize_classification(
    classification_result
):
    """
    Normalize classifier output so the
    API always returns the same schema.
    """

    if not isinstance(
        classification_result,
        dict,
    ):
        return {
            "class": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {},
            "model": (
                "spilltrace_classifier.pth"
            ),
        }

    class_name = (
        classification_result.get(
            "class"
        )
        or classification_result.get(
            "classification"
        )
        or classification_result.get(
            "predicted_class"
        )
        or "UNKNOWN"
    )

    confidence = (
        classification_result.get(
            "confidence"
        )
        or classification_result.get(
            "score"
        )
        or 0.0
    )

    probabilities = (
        classification_result.get(
            "probabilities"
        )
        or {}
    )

    model = (
        classification_result.get(
            "model"
        )
        or "spilltrace_classifier.pth"
    )

    return {
        "class": str(
            class_name
        ).upper(),

        "confidence": float(
            confidence
        ),

        "probabilities": probabilities,

        "model": model,

        "classes": [
            "OIL",
            "LOOKALIKE",
            "NO OIL",
        ],
    }


# ============================================================
# AIS NORMALIZATION
# ============================================================

def get_vessel_coordinates(
    vessel
):
    """
    Accept several possible coordinate
    field names from the AIS prototype.
    """

    if not isinstance(
        vessel,
        dict,
    ):
        return None, None

    latitude = (
        vessel.get("latitude")
        if vessel.get("latitude")
        is not None
        else vessel.get("lat")
    )

    longitude = (
        vessel.get("longitude")
        if vessel.get("longitude")
        is not None
        else vessel.get("lon")
    )

    latitude = safe_float(
        latitude
    )

    longitude = safe_float(
        longitude
    )

    return (
        latitude,
        longitude,
    )


# ============================================================
# FORWARD DRIFT VALIDATION
# ============================================================

def run_forward_drift_validation(
    vessels,
    spill_lat,
    spill_lon,
):
    """
    Run forward drift validation for every
    AIS candidate vessel.
    """

    results = []

    for vessel in vessels:

        if not isinstance(
            vessel,
            dict,
        ):
            continue

        vessel_lat, vessel_lon = (
            get_vessel_coordinates(
                vessel
            )
        )

        if (
            vessel_lat is None
            or vessel_lon is None
        ):

            vessel[
                "forward_drift"
            ] = {
                "validated": False,
                "validation_level": (
                    "UNAVAILABLE"
                ),
                "reason": (
                    "Vessel coordinates "
                    "unavailable"
                ),
            }

            continue

        try:

            validation = (
                validate_forward_drift(
                    vessel_lat=vessel_lat,
                    vessel_lon=vessel_lon,

                    spill_lat=spill_lat,
                    spill_lon=spill_lon,

                    wind_speed_knots=(
                        WIND_SPEED_KNOTS
                    ),

                    wind_from_direction=(
                        WIND_FROM_DIRECTION
                    ),

                    current_speed_knots=(
                        CURRENT_SPEED_KNOTS
                    ),

                    current_to_direction=(
                        CURRENT_TO_DIRECTION
                    ),

                    hours=DRIFT_HOURS,
                )
            )

            vessel[
                "forward_drift"
            ] = validation

            results.append(
                validation
            )

        except Exception as error:

            vessel[
                "forward_drift"
            ] = {
                "validated": False,
                "validation_level": (
                    "ERROR"
                ),
                "reason": str(
                    error
                ),
            }

    return results


# ============================================================
# BACKWARD DRIFT
# ============================================================

def run_backward_drift(
    latitude,
    longitude,
):
    """
    Reconstruct a probable source using
    the prototype backward-drift model.
    """

    try:

        return calculate_backward_drift(
            spill_lat=float(latitude),
            spill_lon=float(longitude),
            wind_speed_knots=WIND_SPEED_KNOTS,
            wind_direction_from=WIND_FROM_DIRECTION,
            current_speed_knots=CURRENT_SPEED_KNOTS,
            current_direction_to=CURRENT_TO_DIRECTION,
            hours=DRIFT_HOURS,
            time_step_hours=DRIFT_TIMESTEP_HOURS,
        )

    except Exception as error:

        return {
            "status": "ERROR",
            "error": str(error),
        }


# ============================================================
# CANDIDATE BUILDER
# ============================================================

def build_candidate(
    region,
    index,
    classification,
    unet_result,
    image_path,
):
    """
    Build one normalized candidate.
    """

    location = region.get(
        "location"
    ) or {}

    latitude = safe_float(
        location.get(
            "latitude"
        )
    )

    longitude = safe_float(
        location.get(
            "longitude"
        )
    )

    spill_id = (
        f"SP-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )

    geometry = {
        "detected": True,

        "area_pixels": (
            region.get(
                "area_pixels",
                0,
            )
        ),

        "largest_contour_area_pixels": (
            region.get(
                "largest_contour_area_pixels",
                0,
            )
        ),

        "bbox": region.get(
            "bbox",
            {},
        ),

        "centroid": region.get(
            "centroid",
            {},
        ),

        "contour_count": (
            region.get(
                "contour_count",
                1,
            )
        ),
    }

    candidate = {
        "candidate_id": (
            f"{spill_id}-{index:02d}"
        ),

        "spill_id": spill_id,

        "classification": (
            classification["class"]
        ),

        "classification_confidence": (
            classification[
                "confidence"
            ]
        ),

        "classification_probabilities": (
            classification[
                "probabilities"
            ]
        ),

        "classifier": classification,

        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },

        "geometry": geometry,

        "detector": {
            "name": (
                "SPILLTRACE U-Net"
            ),

            "model": (
                "spilltrace_unet.pth"
            ),

            "threshold": (
                unet_result.get(
                    "threshold",
                    0.5,
                )
            ),

            "mean_confidence": (
                unet_result.get(
                    "mean_confidence",
                    0.0,
                )
            ),

            "max_confidence": (
                unet_result.get(
                    "max_confidence",
                    0.0,
                )
            ),
        },

        "sar_region_confidence": (
            unet_result.get(
                "mean_confidence",
                0.0,
            )
        ),

        "backward_drift": None,

        "ais_vessels": [],

        "forward_drift": {
            "status": "PENDING",
        },

        "pipeline": {
            "sar_detection": (
                "COMPLETED"
            ),

            "spill_segmentation": (
                "COMPLETED"
            ),

            "spill_geometry": (
                "COMPLETED"
            ),

            "spill_geolocation": (
                "COMPLETED"
            ),

            "lookalike_validation": (
                "COMPLETED"
            ),

            "backward_drift": (
                "PENDING"
            ),

            "ais_correlation": (
                "PENDING"
            ),

            "vessel_ranking": (
                "PENDING"
            ),

            "forward_drift_validation": (
                "PENDING"
            ),
        },
    }

    return candidate


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "SPILLTRACE",

        "status": "running",

        "message": (
            "Satellite-AIS Intelligence "
            "for Oil Spill Source Attribution"
        ),

        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "service": "SPILLTRACE API",

        "timestamp": utc_timestamp(),
    }


# ============================================================
# SYSTEM STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    return {
        "success": True,

        "system": "SPILLTRACE",

        "status": "OPERATIONAL",

        "components": {
            "satellite": (
                "Sentinel-1 SAR"
            ),

            "segmentation": (
                "SPILLTRACE U-Net"
            ),

            "classification": (
                "SPILLTRACE 3-Class "
                "SAR Classifier"
            ),

            "ais": (
                "Synthetic AIS - Prototype"
                if AIS_AVAILABLE
                else "NOT_CONNECTED"
            ),

            "backward_drift": (
                "Synthetic Wind/Current "
                "Prototype"
                if BACKWARD_DRIFT_AVAILABLE
                else "NOT_CONNECTED"
            ),

            "forward_drift": (
                "Synthetic Wind/Current "
                "Prototype"
                if FORWARD_DRIFT_AVAILABLE
                else "NOT_CONNECTED"
            ),
        },
    }


# ============================================================
# DETECT SPILL
# ============================================================

@app.post("/detect-spill")
async def detect_spill(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    }

    filename = (
        file.filename
        or "uploaded_image"
    )

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension not in (
        allowed_extensions
    ):

        return {
            "success": False,

            "error": (
                "Unsupported image format. "
                "Use TIFF, PNG or JPG."
            ),
        }

    file_data = await file.read()

    if not file_data:

        return {
            "success": False,
            "error": "Uploaded file is empty",
        }

    temporary_path = None

    try:

        # ----------------------------------------------------
        # Temporary input file
        # ----------------------------------------------------

        temporary_file = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            )
        )

        temporary_file.write(
            file_data
        )

        temporary_file.close()

        temporary_path = (
            temporary_file.name
        )

        # ----------------------------------------------------
        # TIFF / SAR ML PIPELINE
        # ----------------------------------------------------

        if extension in {
            ".tif",
            ".tiff",
        }:

            # -----------------------------------------------
            # U-NET
            # -----------------------------------------------

            unet_result = (
                detect_spill_unet(
                    temporary_path
                )
            )

            if not unet_result.get(
                "spill_detected",
                False,
            ):

                return {
                    "success": True,

                    "spill_detected": False,

                    "message": (
                        "U-Net did not detect "
                        "a suspicious spill region."
                    ),

                    "detector": {
                        "name": (
                            "SPILLTRACE U-Net"
                        ),

                        "model": (
                            "spilltrace_unet.pth"
                        ),
                    },

                    "pipeline": {
                        "sar_detection": (
                            "COMPLETED"
                        ),

                        "spill_segmentation": (
                            "COMPLETED"
                        ),

                        "spill_geometry": (
                            "COMPLETED"
                        ),

                        "spill_geolocation": (
                            "PENDING"
                        ),

                        "backward_drift": (
                            "PENDING"
                        ),

                        "ais_correlation": (
                            "PENDING"
                        ),

                        "vessel_ranking": (
                            "PENDING"
                        ),

                        "forward_drift_validation": (
                            "PENDING"
                        ),
                    },
                }

            # -----------------------------------------------
            # 3-CLASS CLASSIFIER
            # -----------------------------------------------

            try:

                classifier_raw = (
                    classify_sar_image(
                        temporary_path
                    )
                )

                classification = (
                    normalize_classification(
                        classifier_raw
                    )
                )

            except Exception as error:

                classification = {
                    "class": "UNKNOWN",

                    "confidence": 0.0,

                    "probabilities": {},

                    "model": (
                        "spilltrace_classifier.pth"
                    ),

                    "classes": [
                        "OIL",
                        "LOOKALIKE",
                        "NO OIL",
                    ],

                    "error": str(
                        error
                    ),
                }

            # -----------------------------------------------
            # EXTRACT MULTIPLE U-NET REGIONS
            # -----------------------------------------------

            regions = (
                extract_spill_regions(
                    mask=unet_result.get(
                        "mask"
                    ),

                    image_path=temporary_path,
                )
            )

            # Safety fallback
            if not regions:

                geometry = (
                    unet_result.get(
                        "geometry",
                        {}
                    )
                )

                location = (
                    unet_result.get(
                        "geographic_location"
                    )
                    or {}
                )

                if (
                    geometry.get(
                        "detected"
                    )
                    and location
                ):

                    regions = [
                        {
                            "area_pixels": (
                                geometry.get(
                                    "area_pixels",
                                    0,
                                )
                            ),

                            "largest_contour_area_pixels": (
                                geometry.get(
                                    "largest_contour_area_pixels",
                                    0,
                                )
                            ),

                            "bbox": geometry.get(
                                "bbox",
                                {},
                            ),

                            "centroid": geometry.get(
                                "centroid",
                                {},
                            ),

                            "contour_count": (
                                geometry.get(
                                    "contour_count",
                                    1,
                                )
                            ),

                            "location": location,
                        }
                    ]

            # -----------------------------------------------
            # LIMIT VERY SMALL COMPONENTS
            # -----------------------------------------------

            regions = [
                region
                for region in regions
                if region.get(
                    "area_pixels",
                    0,
                ) >= 500
            ]

            # -----------------------------------------------
            # BUILD CANDIDATES
            # -----------------------------------------------

            candidates = []

            for index, region in enumerate(
                regions,
                start=1,
            ):

                candidate = (
                    build_candidate(
                        region=region,

                        index=index,

                        classification=(
                            classification
                        ),

                        unet_result=(
                            unet_result
                        ),

                        image_path=(
                            temporary_path
                        ),
                    )
                )

                # -------------------------------------------
                # BACKWARD DRIFT
                # -------------------------------------------

                candidate_lat = safe_float(
                    candidate[
                        "location"
                    ].get(
                        "latitude"
                    )
                )

                candidate_lon = safe_float(
                    candidate[
                        "location"
                    ].get(
                        "longitude"
                    )
                )

                if (
                    candidate_lat is not None
                    and candidate_lon is not None
                ):

                    backward_result = (
                        run_backward_drift(
                            candidate_lat,
                            candidate_lon,
                        )
                    )

                    # Keep the complete drift object, but also expose
                    # the important source fields directly on the
                    # candidate. This makes the API schema explicit
                    # and prevents the frontend from depending on
                    # one particular nested field name.
                    candidate[
                        "backward_drift"
                    ] = backward_result

                    if isinstance(backward_result, dict):
                        estimated_source = (
                            backward_result.get(
                                "estimated_source"
                            )
                            or backward_result.get(
                                "source"
                            )
                            or backward_result.get(
                                "origin"
                            )
                        )

                        trajectory = (
                            backward_result.get(
                                "trajectory"
                            )
                            or backward_result.get(
                                "path"
                            )
                            or backward_result.get(
                                "backward_trajectory"
                            )
                            or []
                        )

                        # Fallback: the oldest point in a backward
                        # trajectory is the reconstructed source.
                        if (
                            not estimated_source
                            and isinstance(
                                trajectory,
                                list,
                            )
                            and trajectory
                        ):
                            estimated_source = (
                                trajectory[-1]
                            )

                        if estimated_source:
                            candidate[
                                "estimated_source"
                            ] = estimated_source

                        uncertainty = (
                            backward_result.get(
                                "uncertainty_radius_km"
                            )
                            if isinstance(
                                backward_result,
                                dict,
                            )
                            else None
                        )

                        if uncertainty is None:
                            uncertainty = (
                                backward_result.get(
                                    "source_uncertainty_km"
                                )
                                if isinstance(
                                    backward_result,
                                    dict,
                                )
                                else None
                            )

                        if uncertainty is not None:
                            candidate[
                                "source_uncertainty_km"
                            ] = float(
                                uncertainty
                            )

                        candidate[
                            "backward_drift_status"
                        ] = (
                            "COMPLETED"
                            if (
                                estimated_source
                                or trajectory
                            )
                            else "AVAILABLE"
                        )

                    if (
                        backward_result.get(
                            "status"
                        )
                        != "ERROR"
                    ):

                        candidate[
                            "pipeline"
                        ][
                            "backward_drift"
                        ] = "COMPLETED"

                # -------------------------------------------
                # AIS CORRELATION
                # -------------------------------------------

                ais_vessels = []

                if (
                    AIS_AVAILABLE
                    and
                    candidate_lat
                    is not None
                    and
                    candidate_lon
                    is not None
                ):

                    try:

                        ais_vessels = (
                            correlate_ais(
                                candidate_lat,
                                candidate_lon,
                            )
                        )

                        if not isinstance(
                            ais_vessels,
                            list,
                        ):
                            ais_vessels = []

                        candidate[
                            "pipeline"
                        ][
                            "ais_correlation"
                        ] = "COMPLETED"

                        candidate[
                            "pipeline"
                        ][
                            "vessel_ranking"
                        ] = "COMPLETED"

                    except Exception as error:

                        ais_vessels = []

                        candidate[
                            "pipeline"
                        ][
                            "ais_correlation"
                        ] = (
                            "ERROR"
                        )

                        candidate[
                            "ais_error"
                        ] = str(
                            error
                        )

                # -------------------------------------------
                # FORWARD DRIFT
                # -------------------------------------------

                if (
                    ais_vessels
                    and
                    candidate_lat
                    is not None
                    and
                    candidate_lon
                    is not None
                ):

                    forward_results = (
                        run_forward_drift_validation(
                            vessels=(
                                ais_vessels
                            ),

                            spill_lat=(
                                candidate_lat
                            ),

                            spill_lon=(
                                candidate_lon
                            ),
                        )
                    )

                    candidate[
                        "ais_vessels"
                    ] = ais_vessels

                    candidate[
                        "forward_drift"
                    ] = {
                        "status": (
                            "COMPLETED"
                            if forward_results
                            else "PENDING"
                        ),

                        "hours": (
                            DRIFT_HOURS
                        ),

                        "validation_count": (
                            len(
                                forward_results
                            )
                        ),
                    }

                    if forward_results:

                        candidate[
                            "pipeline"
                        ][
                            "forward_drift_validation"
                        ] = "COMPLETED"

                else:

                    candidate[
                        "ais_vessels"
                    ] = ais_vessels

                candidates.append(
                    candidate
                )

            # -----------------------------------------------
            # PREDICTION MASK
            # -----------------------------------------------

            mask_filename = (
                f"mask_"
                f"{uuid.uuid4().hex[:8]}"
                f".png"
            )

            mask_path = (
                PREDICTION_DIR
                / mask_filename
            )

            save_prediction_mask(
                unet_result[
                    "mask"
                ],

                mask_path,
            )

            # -----------------------------------------------
            # TOP-LEVEL AIS
            # -----------------------------------------------

            all_ais_vessels = []

            if candidates:

                all_ais_vessels = (
                    candidates[0].get(
                        "ais_vessels",
                        [],
                    )
                )

            # -----------------------------------------------
            # PIPELINE STATUS
            # -----------------------------------------------

            any_backward = any(
                candidate[
                    "pipeline"
                ][
                    "backward_drift"
                ]
                == "COMPLETED"
                for candidate
                in candidates
            )

            any_ais = any(
                candidate[
                    "pipeline"
                ][
                    "ais_correlation"
                ]
                == "COMPLETED"
                for candidate
                in candidates
            )

            any_ranking = any(
                candidate[
                    "pipeline"
                ][
                    "vessel_ranking"
                ]
                == "COMPLETED"
                for candidate
                in candidates
            )

            any_forward = any(
                candidate[
                    "pipeline"
                ][
                    "forward_drift_validation"
                ]
                == "COMPLETED"
                for candidate
                in candidates
            )

            # -----------------------------------------------
            # TOP-LEVEL BACKWARD DRIFT
            # -----------------------------------------------

            top_backward_drift = None
            top_estimated_source = None
            top_source_uncertainty_km = None

            if candidates:

                top_backward_drift = (
                    candidates[0].get(
                        "backward_drift"
                    )
                )

                top_estimated_source = (
                    candidates[0].get(
                        "estimated_source"
                    )
                )

                top_source_uncertainty_km = (
                    candidates[0].get(
                        "source_uncertainty_km"
                    )
                )

            # -----------------------------------------------
            # RESPONSE
            # -----------------------------------------------

            response = {

                "success": True,

                "spill_detected": bool(
                    len(candidates) > 0
                ),

                "spill_id": (
                    candidates[0][
                        "spill_id"
                    ]
                    if candidates
                    else generate_spill_id()
                ),

                "timestamp": (
                    utc_timestamp()
                ),

                "detector": {
                    "name": (
                        "SPILLTRACE U-Net"
                    ),

                    "model": (
                        "spilltrace_unet.pth"
                    ),

                    "threshold": float(
                        unet_result.get(
                            "threshold",
                            0.5,
                        )
                    ),

                    "mean_confidence": float(
                        unet_result.get(
                            "mean_confidence",
                            0.0,
                        )
                    ),

                    "max_confidence": float(
                        unet_result.get(
                            "max_confidence",
                            0.0,
                        )
                    ),
                },

                "classification": (
                    classification
                ),

                "candidates": (
                    candidates
                ),

                "detections": (
                    candidates
                ),

                "regions": (
                    candidates
                ),

                "candidate_count": (
                    len(candidates)
                ),

                "location": (
                    candidates[0][
                        "location"
                    ]
                    if candidates
                    else None
                ),

                "geometry": (
                    candidates[0][
                        "geometry"
                    ]
                    if candidates
                    else {}
                ),

                "backward_drift": (
                    top_backward_drift
                ),

                # Explicit top-level source fields for the selected
                # / first candidate. The frontend can use these
                # without unpacking the nested drift object.
                "estimated_source": (
                    top_estimated_source
                ),

                "source_uncertainty_km": (
                    top_source_uncertainty_km
                ),

                "ais": {
                    "available": (
                        AIS_AVAILABLE
                    ),

                    "source": (
                        "Synthetic AIS - Prototype"
                        if AIS_AVAILABLE
                        else "NOT_CONNECTED"
                    ),

                    "vessels": (
                        all_ais_vessels
                    ),
                },

                "wind": {
                    "source": (
                        "Synthetic Prototype"
                    ),

                    "status": (
                        "CONNECTED - PROTOTYPE"
                    ),

                    "speed_knots": (
                        WIND_SPEED_KNOTS
                    ),

                    "wind_from_direction": (
                        WIND_FROM_DIRECTION
                    ),

                    "wind_to_direction": (
                        (
                            WIND_FROM_DIRECTION
                            + 180
                        ) % 360
                    ),
                },

                "current": {
                    "source": (
                        "Synthetic Prototype"
                    ),

                    "status": (
                        "CONNECTED - PROTOTYPE"
                    ),

                    "speed_knots": (
                        CURRENT_SPEED_KNOTS
                    ),

                    "current_to_direction": (
                        CURRENT_TO_DIRECTION
                    ),
                },

                "data_sources": {

                    "satellite": (
                        "Sentinel-1 SAR"
                    ),

                    "ais": (
                        "Synthetic AIS - Prototype"
                    ),

                    "wind": (
                        "Synthetic Wind - Prototype"
                    ),

                    "ocean_current": (
                        "Synthetic Current - Prototype"
                    ),
                },

                "prediction_mask": {

                    "filename": (
                        mask_filename
                    ),

                    "url": (
                        "/prediction-mask/"
                        f"{mask_filename}"
                    ),
                },

                "pipeline": {

                    "sar_detection": (
                        "COMPLETED"
                    ),

                    "spill_segmentation": (
                        "COMPLETED"
                    ),

                    "spill_geometry": (
                        "COMPLETED"
                    ),

                    "spill_geolocation": (
                        "COMPLETED"
                        if candidates
                        else "PENDING"
                    ),

                    "lookalike_validation": (
                        "COMPLETED"
                    ),

                    "backward_drift": (
                        "COMPLETED"
                        if any_backward
                        else "PENDING"
                    ),

                    "ais_correlation": (
                        "COMPLETED"
                        if any_ais
                        else "PENDING"
                    ),

                    "vessel_ranking": (
                        "COMPLETED"
                        if any_ranking
                        else "PENDING"
                    ),

                    "forward_drift_validation": (
                        "COMPLETED"
                        if any_forward
                        else "PENDING"
                    ),
                },

                "notes": [

                    "SPILLTRACE prototype pipeline",

                    "U-Net segmentation is a prototype ML model.",

                    "Classifier probabilities are model outputs, "
                    "not validated real-world probabilities.",

                    "AIS data is synthetic prototype data.",

                    "Wind and ocean-current data are synthetic "
                    "prototype inputs.",

                    "Backward and forward drift are prototype "
                    "reconstructions.",

                    "Forward drift compatibility does not establish "
                    "vessel causality.",
                ],
            }

            return response

        # ----------------------------------------------------
        # NON-TIFF IMAGE
        # ----------------------------------------------------

        else:

            return {
                "success": False,

                "error": (
                    "PNG/JPG processing is not connected "
                    "to the ML SAR pipeline yet. "
                    "Please use the Sentinel-1 TIFF dataset."
                ),
            }

    except Exception as error:

        print(
            "\nSPILLTRACE ANALYSIS ERROR:"
        )

        print(
            repr(error)
        )

        return {
            "success": False,

            "error": str(
                error
            ),

            "error_type": type(
                error
            ).__name__,
        }

    finally:

        if (
            temporary_path
            and os.path.exists(
                temporary_path
            )
        ):

            try:

                os.remove(
                    temporary_path
                )

            except OSError:
                pass


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("=" * 70)
    print(
        "SPILLTRACE API STARTED"
    )
    print("=" * 70)

    print(
        "Satellite: Sentinel-1 SAR"
    )

    print(
        "Segmentation: SPILLTRACE U-Net"
    )

    print(
        "Classification: "
        "SPILLTRACE 3-Class SAR Classifier"
    )

    print(
        "Classes: OIL / LOOKALIKE / NO OIL"
    )

    print(
        "Multiple spill-region detection: ENABLED"
    )

    print(
        "AIS correlation: "
        "CONNECTED - SYNTHETIC PROTOTYPE"
    )

    print(
        "Backward drift: "
        "CONNECTED - SYNTHETIC PROTOTYPE"
    )

    print(
        "Forward drift validation: "
        "CONNECTED - SYNTHETIC PROTOTYPE"
    )

    print(
        "Wind: 8 kn | 270° FROM → 90° TO"
    )

    print(
        "Ocean Current: 1.2 kn | 90° TO"
    )

    print(
        "Forward drift window: 6 hours"
    )

    print("=" * 70)
    print()