import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.types as fot

# Download the COCO 2014 dataset (subset)
dataset = foz.load_zoo_dataset("coco-2014", split="train", max_samples=100)

export_path = "./dataset"

# Export the dataset in COCO format
dataset.export(
    export_dir=export_path,
    dataset_type=fot.COCODetectionDataset,  # Specify dataset type
)

print(f"Dataset exported to {export_path}")
