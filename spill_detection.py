import cv2
import numpy as np
import rasterio

from modules.dataset_loader import load_sentinel_image


# ============================================================
# LOAD SENTINEL-1 VV AS GRAYSCALE
# ============================================================

def load_sentinel_grayscale(image_path):
    """
    Load Sentinel-1 VV channel and convert it to
    an 8-bit grayscale image for OpenCV processing.
    """

    data = load_sentinel_image(image_path)

    vv = data["vv"].astype(np.float32)

    # Remove invalid values
    vv = np.nan_to_num(
        vv,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Robust normalization
    low = np.percentile(vv, 1)
    high = np.percentile(vv, 99)

    if high <= low:
        high = low + 1.0

    vv = np.clip(
        vv,
        low,
        high
    )

    normalized = (
        (vv - low) /
        (high - low)
    )

    grayscale = (
        normalized * 255
    ).astype(np.uint8)

    return grayscale


# ============================================================
# PIXEL COORDINATE → REAL LATITUDE / LONGITUDE
# ============================================================

def pixel_to_latlon(image_path, x, y):
    """
    Convert Sentinel-1 image pixel coordinates into
    real geographic latitude and longitude.

    x = column / horizontal pixel coordinate
    y = row / vertical pixel coordinate

    The Sentinel-1 TIFF must contain valid geospatial
    transform information.
    """

    with rasterio.open(image_path) as src:

        lon, lat = rasterio.transform.xy(
            src.transform,
            y,
            x,
            offset="center"
        )

    return {
        "latitude": float(lat),
        "longitude": float(lon)
    }


# ============================================================
# DETECT SUSPICIOUS DARK REGIONS
# ============================================================

def detect_dark_regions(image_path):
    """
    Detect suspicious dark regions in a Sentinel-1 SAR image.

    This is a rule-based SAR candidate detector.
    It does NOT independently confirm that a region is oil.
    """

    image = load_sentinel_grayscale(
        image_path
    )

    image_height, image_width = image.shape

    image_area = (
        image_height *
        image_width
    )

    # --------------------------------------------------------
    # Reduce SAR speckle
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image,
        (7, 7),
        0
    )

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    image_mean = float(
        np.mean(blurred)
    )

    image_median = float(
        np.median(blurred)
    )

    image_std = float(
        np.std(blurred)
    )

    # --------------------------------------------------------
    # Detect unusually dark pixels
    #
    # Using the 12th percentile instead of 20th percentile
    # reduces the number of ordinary dark ocean regions.
    # --------------------------------------------------------

    threshold_value = np.percentile(
        blurred,
        12
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # --------------------------------------------------------
    # Morphological cleaning
    # --------------------------------------------------------

    open_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    close_kernel = np.ones(
        (7, 7),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    # --------------------------------------------------------
    # Process every contour
    # --------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        area_ratio = (
            area /
            image_area
        )

        # ----------------------------------------------------
        # Ignore extremely small regions
        # ----------------------------------------------------

        if area_ratio < 0.0005:
            continue

        # ----------------------------------------------------
        # Ignore extremely large regions
        #
        # Very large connected dark regions are often
        # background / imaging artifacts rather than
        # compact spill candidates.
        # ----------------------------------------------------

        if area_ratio > 0.08:
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, width, height = cv2.boundingRect(
            contour
        )

        # ----------------------------------------------------
        # Ignore image boundaries
        # ----------------------------------------------------

        touches_boundary = (
            x <= 0 or
            y <= 0 or
            x + width >= image_width - 1 or
            y + height >= image_height - 1
        )

        if touches_boundary:
            continue

        # ----------------------------------------------------
        # Perimeter
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        aspect_ratio = (
            max(width, height) /
            max(min(width, height), 1)
        )

        # ----------------------------------------------------
        # Compactness
        # ----------------------------------------------------

        compactness = (
            4 * np.pi * area
        ) / (
            perimeter ** 2
        )

        # ----------------------------------------------------
        # Candidate mask
        # ----------------------------------------------------

        mask = np.zeros_like(
            image
        )

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            -1
        )

        region_pixels = image[
            mask == 255
        ]

        if len(region_pixels) == 0:
            continue

        # ----------------------------------------------------
        # Mean backscatter
        # ----------------------------------------------------

        mean_backscatter = float(
            np.mean(region_pixels)
        )

        # ----------------------------------------------------
        # Darkness score
        # ----------------------------------------------------

        darkness_difference = (
            image_median -
            mean_backscatter
        )

        if image_std > 0:

            darkness_score = (
                darkness_difference /
                image_std
            )

        else:

            darkness_score = 0.0

        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------

        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:

            centroid_x = (
                moments["m10"] /
                moments["m00"]
            )

            centroid_y = (
                moments["m01"] /
                moments["m00"]
            )

        else:

            centroid_x = (
                x +
                width / 2
            )

            centroid_y = (
                y +
                height / 2
            )

        # ----------------------------------------------------
        # REAL GEOGRAPHIC LOCATION
        # ----------------------------------------------------

        geo = pixel_to_latlon(
            image_path,
            centroid_x,
            centroid_y
        )

        latitude = geo["latitude"]
        longitude = geo["longitude"]

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        rect = cv2.minAreaRect(
            contour
        )

        orientation = float(
            rect[2]
        )

        # ----------------------------------------------------
        # Candidate quality score
        #
        # This is a ranking score, NOT an oil probability.
        # ----------------------------------------------------

        candidate_score = 0.0

        # Darkness
        if darkness_score >= 1.5:

            candidate_score += 35

        elif darkness_score >= 0.8:

            candidate_score += 25

        elif darkness_score >= 0.4:

            candidate_score += 10

        # Elongated geometry
        if 1.3 <= aspect_ratio <= 8:

            candidate_score += 25

        elif aspect_ratio <= 12:

            candidate_score += 10

        # Reasonable area
        if 0.001 <= area_ratio <= 0.04:

            candidate_score += 20

        elif area_ratio <= 0.08:

            candidate_score += 5

        # Irregular / moderate compactness
        if 0.05 <= compactness <= 0.65:

            candidate_score += 20

        elif compactness <= 0.85:

            candidate_score += 5

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        candidates.append({

            "area_pixels": float(
                area
            ),

            "area_ratio": float(
                area_ratio
            ),

            "perimeter_pixels": float(
                perimeter
            ),

            # Pixel coordinates
            "centroid_x": float(
                centroid_x
            ),

            "centroid_y": float(
                centroid_y
            ),

            # Real geographic coordinates
            "latitude": round(
                latitude,
                6
            ),

            "longitude": round(
                longitude,
                6
            ),

            "width_pixels": int(
                width
            ),

            "height_pixels": int(
                height
            ),

            "aspect_ratio": round(
                float(aspect_ratio),
                3
            ),

            "compactness": round(
                float(compactness),
                3
            ),

            "mean_backscatter": round(
                mean_backscatter,
                3
            ),

            "darkness_score": round(
                float(darkness_score),
                3
            ),

            "orientation": round(
                orientation,
                3
            ),

            "candidate_score": round(
                candidate_score,
                1
            )
        })

    # --------------------------------------------------------
    # Sort by candidate quality
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item:
            item["candidate_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Keep the strongest candidates
    #
    # This prevents the frontend from being flooded
    # with weak dark-region detections.
    # --------------------------------------------------------

    candidates = candidates[:7]

    # --------------------------------------------------------
    # Return analysis
    # --------------------------------------------------------

    return {

        "image_width": int(
            image_width
        ),

        "image_height": int(
            image_height
        ),

        "image_statistics": {

            "mean": round(
                image_mean,
                3
            ),

            "median": round(
                image_median,
                3
            ),

            "std": round(
                image_std,
                3
            ),

            "threshold": round(
                float(threshold_value),
                3
            )
        },

        "candidate_count": len(
            candidates
        ),

        "candidates": candidates
    }


# ============================================================
# CREATE ANNOTATED SAR IMAGE
# ============================================================

def create_annotated_image(
    image_path,
    output_path,
    candidates=None
):
    """
    Create an annotated Sentinel-1 SAR image.

    The candidates supplied by the analysis pipeline are
    reused so the displayed boxes match the analysis.
    """

    image = load_sentinel_grayscale(
        image_path
    )

    # --------------------------------------------------------
    # Run detection if candidates were not supplied
    # --------------------------------------------------------

    if candidates is None:

        detection = detect_dark_regions(
            image_path
        )

        candidates = detection[
            "candidates"
        ]

    # --------------------------------------------------------
    # Convert grayscale to BGR
    # --------------------------------------------------------

    annotated = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Draw candidates
    # --------------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        center_x = int(
            candidate["centroid_x"]
        )

        center_y = int(
            candidate["centroid_y"]
        )

        width = int(
            candidate["width_pixels"]
        )

        height = int(
            candidate["height_pixels"]
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1 = max(
            0,
            center_x - width // 2
        )

        y1 = max(
            0,
            center_y - height // 2
        )

        x2 = min(
            image.shape[1] - 1,
            center_x + width // 2
        )

        y2 = min(
            image.shape[0] - 1,
            center_y + height // 2
        )

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Draw centroid
        # ----------------------------------------------------

        cv2.circle(
            annotated,
            (center_x, center_y),
            5,
            (255, 0, 0),
            -1
        )

        # ----------------------------------------------------
        # Candidate label
        # ----------------------------------------------------

        label = (
            f"C{index} | "
            f"Score: "
            f"{candidate['candidate_score']:.0f}"
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )[0]

        text_width = text_size[0]
        text_height = text_size[1]

        label_x = x1

        label_y = max(
            text_height + 5,
            y1 - 5
        )

        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (
                label_x,
                label_y - text_height - 5
            ),
            (
                label_x +
                text_width +
                6,
                label_y + 2
            ),
            (0, 0, 0),
            -1
        )

        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            label,
            (
                label_x + 3,
                label_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = (
        f"SPILLTRACE - "
        f"{len(candidates)} SAR Candidate(s)"
    )

    cv2.putText(
        annotated,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    success = cv2.imwrite(
        output_path,
        annotated
    )

    return success