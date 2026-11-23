# Team 16 | Skin lesion classification on HAM10000

Course project for Introduction to Data Science for Engineers at IIITDM Kancheepuram. We compared nine model configurations on a lesion-grouped HAM10000 split and used macro-F1 as the primary measure of class-imbalanced performance. The selected ConvNeXt-Tiny checkpoint achieved test macro-F1 of **0.7783**, accuracy of **0.8346**, and balanced accuracy of **0.8123**.

## Submission

| Artifact | Location |
| --- | --- |
| Final report | [Team16_Report.pdf](report/Team16_Report.pdf) |
| Report source | [main.tex](report/main.tex), [references.bib](report/references.bib), [figures](report/figures/) |
| Literature survey | [literature_survey.pdf](survey/literature_survey.pdf) |
| Dataset and split setup | [data/README.md](data/README.md) |
| Training and Grad-CAM notebooks | [notebooks/README.md](notebooks/README.md) |
| Saved results, prediction arrays, and verification | [results/README.md](results/README.md) |
| Web app setup | [webapp/README.md](webapp/README.md) |
| Contributions | [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |

## Reproduce the saved evaluation

From the repository root, run:

~~~bash
python -m pip install -r requirements.txt
python results/recompute.py
~~~

This recalculates metrics from the saved predictions, including the selected model's class-level results and calibration. Follow the [results instructions](results/README.md) to regenerate the confusion matrix, reliability diagram, and coverage curve. This check uses saved predictions and runs without the raw images or the model checkpoint.

The [notebook index](notebooks/README.md) lists the included experiments. The original lesion-grouped split CSVs are in [data/splits](data/splits/). See the [dataset guide](data/README.md) for data setup.

## Web demo

The [live demo](https://skin-lesion-classification-umber.vercel.app/) accepts a lesion image and shows class scores, an optional Grad-CAM overlay, and a downloadable PDF summary. See the [web app README](webapp/README.md) for local setup and checkpoint placement. This is an educational research project. Its output is not a medical diagnosis.
