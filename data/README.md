# HAM10000 data

Download the two HAM10000 image archives and the metadata table from the [official Harvard Dataverse record](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). Use the original CSV version of the metadata. The dataset record includes its citation and terms of use.

For the included Colab notebooks, extract the archives into distinct image folders and set the dataset path in the setup cell. The training notebook expects the metadata as `HAM10000_metadata.csv` and the image folders as `HAM10000_images_part_1` and `HAM10000_images_part_2`. The Grad-CAM notebook takes a list of image directories in its setup cell. Keep the downloaded data outside Git, for example under `data/raw/` locally.

## Project splits

The original project manifests are [train_split.csv](splits/train_split.csv), [val_split.csv](splits/val_split.csv), and [test_split.csv](splits/test_split.csv). They contain 7,991 training images, 1,008 validation images, and 1,016 test images before training-only near-duplicate removal. There are no shared lesion or image IDs across the three partitions.

Point each notebook's split-directory setting at `data/splits/` or copy these files to the expected location in Colab. Keep their original row order. The validation and test class labels match the labels in the saved prediction arrays row by row.
