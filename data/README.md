# HAM10000 data

Download the two HAM10000 image archives and the metadata table from the [official Harvard Dataverse record](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). Use the original CSV version of the metadata. The dataset record includes its citation and terms of use.

For the included Colab notebooks, extract the archives into distinct image folders and set the dataset path in the setup cell. The training notebook expects the metadata as `HAM10000_metadata.csv` and the image folders as `HAM10000_images_part_1` and `HAM10000_images_part_2`. The Grad-CAM notebook takes a list of image directories in its setup cell. Keep the downloaded data outside Git, for example under `data/raw/` locally.

## Project splits

The report uses lesion-grouped partitions of 7,991 training images, 1,008 validation images, and 1,016 test images before the training-only near-duplicate removal. The original project CSVs are pending. The submission layout reserves `data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv` for those manifests. The Colab notebooks expect `train_split.csv`, `val_split.csv`, and `test_split.csv` inside their configured split directory.

Preserve the original `image_id`, `lesion_id`, labels, and row order. The saved prediction arrays contain labels and class scores, so they can reproduce the metrics without the images. Pairing an array row with a particular image for Grad-CAM requires the original split order.
