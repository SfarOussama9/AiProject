# AI Project: Bankruptcy Prediction Notebook

This repository contains a complete notebook-based workflow for predicting company bankruptcy from financial indicators.

## Main file

- `bankruptcy_prediction_final.ipynb`: the full analysis, modeling, evaluation, and final inference notebook.

## Repository contents

- `data.csv`: dataset used by the notebook.
- `PROJECT_PLAN.md`: project planning notes.
- `final_metrics.csv`: saved final evaluation metrics for the selected model.
- `bankruptcy_champion.joblib`: exported trained champion pipeline.

## Prerequisites

You need:

- Python 3.11 or newer
- VS Code with the Jupyter extension, or Jupyter Notebook / JupyterLab

## Recommended setup on Windows PowerShell

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install notebook jupyterlab pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost scipy joblib
```

If PowerShell blocks activation, run this once for the current terminal session and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## How to run the notebook in VS Code

1. Open this folder in VS Code.
2. Open `bankruptcy_prediction_final.ipynb`.
3. Select the Python kernel from `.venv` when VS Code asks for a kernel.
4. Run the notebook from top to bottom.
5. For a completely fresh run, use `Restart Kernel and Run All`.

## How to run the notebook in Jupyter

After activating the virtual environment, start Jupyter with one of these commands:

```powershell
jupyter notebook
```

or

```powershell
jupyter lab
```

Then open `bankruptcy_prediction_final.ipynb` and run all cells in order.

## Important notes before running

- Run the cells in order from top to bottom. Later sections depend on variables created earlier.
- The notebook includes repeated cross-validation and randomized hyperparameter search, so a full rerun can take several minutes.
- Keep `data.csv` in the root folder next to the notebook.

## Expected outputs

When the notebook finishes successfully, you should have:

- evaluation tables and plots inside the notebook
- `final_metrics.csv`
- `bankruptcy_champion.joblib`

## Quick rerun checklist

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Then open the notebook and run all cells.