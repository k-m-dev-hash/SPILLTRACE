import rasterio
from rasterio.transform import xy


def get_pixel_coordinates(image_path, pixel_x, pixel_y):
    """
    Convert image pixel coordinates to geographic coordinates.

    pixel_x = image column
    pixel_y = image row

    Returns longitude and latitude.
    """

    with rasterio.open(image_path) as src:

        longitude, latitude = xy(
            src.transform,
            pixel_y,
            pixel_x
        )

        return {
            "latitude": float(latitude),
            "longitude": float(longitude)
        }


def get_image_geolocation(image_path):
    """
    Return the CRS and geographic bounds
    of the Sentinel-1 GeoTIFF.
    """

    with rasterio.open(image_path) as src:

        bounds = src.bounds

        return {
            "crs": str(src.crs),

            "bounds": {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top)
            }
        }