# Experiment notebooks

| Notebook | Contents |
| --- | --- |
| [ConvNeXt-Tiny v5](convnext_tiny_v5/convnext_tiny_v5.ipynb) | Backbone comparison and training pipeline. The saved configuration selects ConvNeXt-Tiny with seed 42. The notebook also has EfficientNet-B3 and ResNet-50 presets. |
| [Grad-CAM and error analysis](gradcam_error_analysis.ipynb) | Grad-CAM, Grad-CAM++, error selection, and gallery generation for the selected ConvNeXt-Tiny run. |

The historical baseline notebooks, the ResNet-50 imbalance progression, and the remaining complex-model training notebooks are pending. The report and saved result files document the available experiment outcomes.

## Running on Colab

1. Download HAM10000 and arrange the files as described in the [data instructions](../data/README.md). Set `DATA_DIR` to the folder containing `HAM10000_metadata.csv` and the two image folders.
2. Set `SPLIT_DIR` to the folder containing the original lesion-grouped split files. The notebooks expect `train_split.csv`, `val_split.csv`, and `test_split.csv`. If your copies use the shorter names `train.csv`, `val.csv`, and `test.csv`, rename them for the notebook while preserving their row order. Leave `RUN_REGROUP` disabled when reproducing the reported training run.
3. Set `OUT_DIR` to a writable experiment folder. The training notebook writes its checkpoint, results, and prediction arrays there. Use a GPU runtime for the full training run.
4. For the Grad-CAM notebook, place the selected `best_v5_convnext_tiny224*seed42.pth` checkpoint in `OUT_DIR`. Copy the [saved prediction archive](../results/convnext_tiny_v5_probs.npz) there as `probs_v5_convnext_tiny224_seed42.npz` so its filename matches the notebook's glob. Use the original test split in the same order as the saved prediction rows. Set `GRADCAM_DIR` to a writable folder for generated figures.

The Grad-CAM notebook retains its code and text outputs. Cached images and Colab account metadata were removed from the submission copy. The generated plots selected for the paper are in [the report figures folder](../report/figures/).
