import os
import json
from pathlib import Path

from modules.spill_detection import detect_dark_regions
from modules.lookalike import validate_candidate


# ============================================================
# SPILLTRACE - FULL DATASET BATCH PROCESSOR
# ============================================================

DATASET_ROOT = Path(
    os.path.expanduser(
        r"~\Downloads\02_Test_images_and_ground_truth"
    )
)

IMAGES_ROOT = DATASET_ROOT / "Images"

OUTPUT_FILE = Path("batch_results.json")


def get_all_images():
    """
    Find every TIFF image inside:
        Images/Oil
        Images/Lookalike
        Images/No oil
    """

    image_files = []

    for category in ["Oil", "Lookalike", "No oil"]:

        category_folder = IMAGES_ROOT / category

        if not category_folder.exists():
            print(f"WARNING: Folder not found: {category_folder}")
            continue

        files = list(category_folder.rglob("*.tif"))
        files += list(category_folder.rglob("*.tiff"))

        for file in files:
            image_files.append(
                {
                    "path": file,
                    "category": category
                }
            )

    return image_files


def process_image(image_info):

    image_path = image_info["path"]
    actual_category = image_info["category"]

    try:

        # ----------------------------------------------------
        # STEP 1: Detect dark regions
        # ----------------------------------------------------

        detection = detect_dark_regions(str(image_path))

        candidates = detection.get("candidates", [])

        # ----------------------------------------------------
        # STEP 2: Look-alike validation
        # ----------------------------------------------------

        validated_candidates = []

        for candidate in candidates:

            validation = validate_candidate(
                candidate,
                detection.get("statistics", {}),
                wind_speed=None
            )

            validated_candidates.append(
                {
                    "candidate": candidate,
                    "validation": validation
                }
            )

        # ----------------------------------------------------
        # STEP 3: Determine best candidate
        # ----------------------------------------------------

        best_candidate = None

        if validated_candidates:

            best_candidate = max(
                validated_candidates,
                key=lambda x: x["validation"]["validation_score"]
            )

        # ----------------------------------------------------
        # STEP 4: Create result
        # ----------------------------------------------------

        result = {

            "filename": image_path.name,

            "filepath": str(image_path),

            "actual_category": actual_category,

            "status": "success",

            "candidate_count": len(candidates),

            "image_width": detection.get("image_width"),

            "image_height": detection.get("image_height"),

            "best_candidate": best_candidate,

            "all_candidates": validated_candidates
        }

        return result

    except Exception as error:

        return {

            "filename": image_path.name,

            "filepath": str(image_path),

            "actual_category": actual_category,

            "status": "error",

            "error": str(error)
        }


def main():

    print()
    print("=" * 70)
    print("SPILLTRACE - FULL DATASET PROCESSING")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET_ROOT)

    print()
    print("Searching for Sentinel-1 TIFF images...")

    image_files = get_all_images()

    print()
    print(f"Total images found: {len(image_files)}")

    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------

    category_counts = {
        "Oil": 0,
        "Lookalike": 0,
        "No oil": 0
    }

    for item in image_files:
        category_counts[item["category"]] += 1

    print()
    print("Dataset distribution:")
    print(f"  Oil       : {category_counts['Oil']}")
    print(f"  Lookalike : {category_counts['Lookalike']}")
    print(f"  No oil    : {category_counts['No oil']}")

    print()
    print("-" * 70)

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    results = []

    total = len(image_files)

    for index, image_info in enumerate(image_files, start=1):

        print(
            f"[{index}/{total}] "
            f"{image_info['category']} "
            f"-> {image_info['path'].name}"
        )

        result = process_image(image_info)

        results.append(result)

        if result["status"] == "success":

            print(
                f"    Candidates detected: "
                f"{result['candidate_count']}"
            )

            if result["best_candidate"]:

                validation = result["best_candidate"]["validation"]

                print(
                    f"    Classification: "
                    f"{validation['classification']}"
                )

                print(
                    f"    Validation score: "
                    f"{validation['validation_score']}"
                )

            else:

                print("    No candidate detected.")

        else:

            print(
                f"    ERROR: {result['error']}"
            )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    successful = sum(
        1
        for result in results
        if result["status"] == "success"
    )

    errors = sum(
        1
        for result in results
        if result["status"] == "error"
    )

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print()
    print(f"Total images : {total}")
    print(f"Successful   : {successful}")
    print(f"Errors       : {errors}")

    print()
    print(f"Results saved to:")
    print(OUTPUT_FILE.absolute())

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()