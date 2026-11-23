# Experiment notebooks

| Notebook | Contents |
| --- | --- |
| [Plain ResNet-50](baselines/Baseline_Resnet__IDSE__2_.ipynb) | ImageNet-pretrained baseline used as the first step of the imbalance ladder. |
| [EfficientNet-B0](baselines/skin_lesion_efficientnet.ipynb) | Class-weighted EfficientNet-B0 baseline. |
| [Keras custom CNN](baselines/DenseNet121_HAM10000_Baseline.ipynb) | The saved model is a Keras custom CNN. The file keeps its original name. |
| [PyTorch CustomCNN](baselines/base0%20%282%29.ipynb) | Custom convolutional baseline with saved test accuracy 0.5236 and macro-F1 0.3757. |
| [ResNet-50 class weights](resnet50_imbalance/HAM10000_ResNet50_ClassWeights__1_.ipynb) | Weighted-training code. The saved output is a dry run. The full-run score appears in the v3 and v4 comparison tables. |
| [ResNet-50 resampling and deduplication](resnet50_imbalance/v3_HAM10000_ResNet50_Resampling__1_.ipynb) | Training-only deduplication and per-epoch resampling. |
| [ResNet-50 v4](resnet50_imbalance/v4-seed42.ipynb) | Full recipe with the four seed scores, validation-fitted offsets, and the ensemble result. |
| [ConvNeXt-Tiny v5 and EfficientNet-B3](convnext_tiny_v5/convnext_tiny_v5.ipynb) | Shared training notebook. The saved selection is the final ConvNeXt-Tiny experiment. The EfficientNet-B3 preset uses its own saved evaluation files. |
| [Swin-T and shared-recipe TransNet](complex_models/HAM10000_v5_reference_swin_transnet.ipynb) | Training code for both backbone comparisons. |
| [Older paper-recipe TransNet](complex_models/transnet_colab_fixed_v4.ipynb) | Colab notebook configured for the 42-epoch run. Its full-run export matches the filename and structure of the [saved results](../results/results_FULL.json). |
| [EfficientNet-B3 + Swin-T hybrid](complex_models/HAM10000_Hybrid_EfficientNetB3_SwinTiny_MacroF1.ipynb) | Feature fusion with raw and validation-offset results. |
| [Mixup and CutMix experiments](complex_models/HAM10000_v6_boost.ipynb) | Training and inference variants discussed among the unsuccessful experiments. |
| [Metadata fusion](complex_models/skin_lesion_efficientnet_class_imbalance%20%282%29.ipynb) | EfficientNet ensemble with image and metadata inputs, evaluated in the unsuccessful variants. |
| [Grad-CAM and error analysis](gradcam_error_analysis.ipynb) | Grad-CAM, Grad-CAM++, error selection, and gallery generation for the selected ConvNeXt-Tiny run. |

The seed-42 paper-recipe TransNet run uses [train_transnet.py](../scripts/train_transnet.py). Set `DATA_DIR`, `SPLIT_DIR`, and `OUT_DIR` in the script for your machine. For a local run, set `COLAB = False` and run `python scripts/train_transnet.py` from the repository root.

## Running on Colab

1. Download HAM10000 and arrange the files as described in the [data instructions](../data/README.md). Set `DATA_DIR` to the folder containing `HAM10000_metadata.csv` and the two image folders.
2. Set `SPLIT_DIR` to the directory containing the [original split files](../data/splits/). Some notebooks read the splits alongside the images and metadata. Keep the filenames and row order. Leave `RUN_REGROUP` disabled when reproducing the reported training run.
3. Set `OUT_DIR` to a writable experiment folder. The training notebook writes its checkpoint, results, and prediction arrays there. Use a GPU runtime for the full training run.
4. For the Grad-CAM notebook, place the selected `best_v5_convnext_tiny224*seed42.pth` checkpoint in `OUT_DIR`. Copy the [saved prediction archive](../results/convnext_tiny_v5_probs.npz) there as `probs_v5_convnext_tiny224_seed42.npz` so its filename matches the notebook's glob. Use the original test split in the same order as the saved prediction rows. Set `GRADCAM_DIR` to a writable folder for generated figures.

The submission copies retain the experiment code and recorded metrics. Colab account metadata was removed from the newly added notebooks. The Grad-CAM submission copy also omits cached images. The generated plots selected for the paper are in [the report figures folder](../report/figures/).
