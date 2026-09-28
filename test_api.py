import urllib.request
import uuid
import json

IMAGE_PATH = r"C:\Users\Khushi Sharma\Downloads\02_Test_images_and_ground_truth\Images\Oil\00077.tif"
API_URL = "http://127.0.0.1:8000/detect-spill"

with open(IMAGE_PATH, "rb") as f:
    image_data = f.read()

boundary = "----SPILLTRACE" + uuid.uuid4().hex

body = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="file"; filename="00077.tif"\r\n'
    "Content-Type: image/tiff\r\n"
    "\r\n"
).encode() + image_data + (
    f"\r\n--{boundary}--\r\n"
).encode()

request = urllib.request.Request(
    API_URL,
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    },
    method="POST",
)

with urllib.request.urlopen(request) as response:
    result = json.loads(response.read().decode())

print("\n" + "=" * 70)
print("SPILLTRACE API BACKWARD DRIFT CHECK")
print("=" * 70)

print("\nTOP LEVEL ESTIMATED SOURCE:")
print(json.dumps(result.get("estimated_source"), indent=2))

print("\nTOP LEVEL BACKWARD DRIFT:")
print(json.dumps(result.get("backward_drift"), indent=2))

candidates = result.get("candidates", [])

print("\nNUMBER OF CANDIDATES:", len(candidates))

if candidates:
    candidate = candidates[0]

    print("\nFIRST CANDIDATE ESTIMATED SOURCE:")
    print(json.dumps(candidate.get("estimated_source"), indent=2))

    print("\nFIRST CANDIDATE BACKWARD DRIFT:")
    print(json.dumps(candidate.get("backward_drift"), indent=2))

print("\n" + "=" * 70)