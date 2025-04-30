import fiftyone.zoo as foz
import fiftyone as fo
from PIL import Image

# Allow PIL to load incomplete images if needed (optional workaround)
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # does not load full image but checks header/integrity
        return True
    except Exception as e:
        print(f"Invalid image: {path} ({e})")
        return False


# Load dataset
dataset = foz.load_zoo_dataset("coco-2014", split="train", max_samples=4000)

# Track bad sample IDs
bad_ids = []

for sample in dataset:
    if not is_valid_image(sample.filepath):
        bad_ids.append(sample.id)

# Delete bad samples
if bad_ids:
    print(f"Deleting {len(bad_ids)} bad images...")
    dataset.delete_samples(bad_ids)
else:
    print("No corrupted images found.")

# (Optional) launch the app to view what's left
# session = fo.launch_app(dataset)
