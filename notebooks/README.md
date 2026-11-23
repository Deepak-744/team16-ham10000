# Experiment notebooks

| Notebook | Contents |
| --- | --- |
| [Plain ResNet-50](baselines/Baseline_Resnet__IDSE__2_.ipynb) | ImageNet-pretrained baseline used as the first step of the imbalance ladder. |
| [EfficientNet-B0](baselines/skin_lesion_efficientnet.ipynb) | Class-weighted EfficientNet-B0 baseline. |
| [Keras custom CNN](baselines/DenseNet121_HAM10000_Baseline.ipynb) | The saved model is a Keras custom CNN. The file keeps its original name. |
| [ResNet-50 class weights](resnet50_imbalance/HAM10000_ResNet50_ClassWeights__1_.ipynb) | Weighted-training code. The saved output is a dry run. The full-run score appears in the v3 and v4 comparison tables. |
| [ResNet-50 resampling and deduplication](resnet50_imbalance/v3_HAM10000_ResNet50_Resampling__1_.ipynb) | Training-only deduplication and per-epoch resampling. |
| [ResNet-50 v4](resnet50_imbalance/v4-seed42.ipynb) | Full recipe with the four seed scores, validation-fitted offsets, and the ensemble result. |
| [ConvNeXt-Tiny v5 and EfficientNet-B3](convnext_tiny_v5/convnext_tiny_v5.ipynb) | Shared training notebook. The saved selection is the final ConvNeXt-Tiny experiment. The EfficientNet-B3 preset uses its own saved evaluation files. |
| [Swin-T and shared-recipe TransNet](complex_models/HAM10000_v5_reference_swin_transnet.ipynb) | Training code for both backbone comparisons. |
| [EfficientNet-B3 + Swin-T hybrid](complex_models/HAM10000_Hybrid_EfficientNetB3_SwinTiny_MacroF1.ipynb) | Feature fusion with raw and validation-offset results. |
| [Mixup and CutMix experiments](complex_models/HAM10000_v6_boost.ipynb) | Training and inference variants discussed among the unsuccessful experiments. |
| [Grad-CAM and error analysis](gradcam_error_analysis.ipynb) | Grad-CAM, Grad-CAM++, error selection, and gallery generation for the selected ConvNeXt-Tiny run. |

The seed-42 paper-recipe TransNet run uses [train_transnet.py](../scripts/train_transnet.py). Set `DATA_DIR`, `SPLIT_DIR`, and `OUT_DIR` in the script for your machine. For a local run, set `COLAB = False` and run `python scripts/train_transnet.py` from the repository root.

The PyTorch CustomCNN training notebook, metadata-fusion notebook, and source notebook for the older paper-recipe TransNet run are pending. The older TransNet results are available in [the saved results](../results/README.md).

## Running on Colab

1. Download HAM10000 and arrange the files as described in the [data instructions](../data/README.md). Set `DATA_DIR` to the folder containing `HAM10000_metadata.csv` and the two image folders.
2. Set `SPLIT_DIR` to the folder containing the original lesion-grouped split files. The notebooks expect `train_split.csv`, `val_split.csv`, and `test_split.csv`. If your copies use the shorter names `train.csv`, `val.csv`, and `test.csv`, rename them for the notebook while preserving their row order. Leave `RUN_REGROUP` disabled when reproducing the reported training run.
3. Set `OUT_DIR` to a writable experiment folder. The training notebook writes its checkpoint, results, and prediction arrays there. Use a GPU runtime for the full training run.
4. For the Grad-CAM notebook, place the selected `best_v5_convnext_tiny224*seed42.pth` checkpoint in `OUT_DIR`. Copy the [saved prediction archive](../results/convnext_tiny_v5_probs.npz) there as `probs_v5_convnext_tiny224_seed42.npz` so its filename matches the notebook's glob. Use the original test split in the same order as the saved prediction rows. Set `GRADCAM_DIR` to a writable folder for generated figures.

The submission copies retain the experiment code and recorded metrics. Colab account metadata was removed from the newly added notebooks. The Grad-CAM submission copy also omits cached images. The generated plots selected for the paper are in [the report figures folder](../report/figures/).
