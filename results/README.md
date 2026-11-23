# Saved evaluation results

| Files | Experiment |
| --- | --- |
| `convnext_tiny_v5_results.json`, `convnext_tiny_v5_probs.npz` | Selected ConvNeXt-Tiny v5, seed 42, epoch 20 |
| `efficientnet_b3_v5_results.json`, `efficientnet_b3_v5_probs.npz` | EfficientNet-B3 comparison |
| `swin_t_v5_results.json`, `swin_t_v5_probs.npz` | Swin-T comparison |
| `transnet_shared_recipe_v5_results.json`, `transnet_shared_recipe_v5_probs.npz` | TransNet with the shared training recipe |
| `results_transnet_ep42_seed42.json`, `probs_transnet_ep42_seed42.npz` | Seed-42 TransNet paper-recipe reproduction |
| `results_FULL.json` | Earlier TransNet paper-recipe run with recorded aggregate results |
| `figures/` | ConvNeXt test confusion matrix, reliability diagram, and confidence coverage curve from the final report |

The NumPy archives contain `val_probs`, `val_y`, `test_probs`, and `test_y`. They hold 1,008 validation rows and 1,016 test rows. The probability columns and integer class labels follow this order: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. The labels are paired with scores by row. Image and lesion IDs are absent from the archives, so matching a row to an image requires the original split files in evaluation order.

## Recompute the metrics

From the repository root, with NumPy and Matplotlib installed:

```sh
python results/recompute.py
python results/recompute.py --output-dir ../team16-recomputed-results
```

The first command checks the ConvNeXt and seed-42 TransNet paper-recipe predictions against their recorded accuracy, macro-F1, balanced accuracy, and confusion matrices where available. The second writes metrics, including calibration bins and coverage thresholds, and regenerates the three ConvNeXt figures in the specified directory. These checks use saved predictions and do not require the images or model weights.

ConvNeXt validation scores in the archive come from the final flip-averaged evaluation. Its reported best validation macro-F1 comes from the single-view checkpoint selection pass. The older TransNet run is reported from `results_FULL.json`, which has no accompanying per-image predictions.
