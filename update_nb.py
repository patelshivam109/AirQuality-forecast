import json

nb_path = 'notebooks/05_Model_Training.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the index of the "Model Comparison" code cell
target_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and '## 5. Select Best Model and Save\n' in cell['source']:
        target_idx = i
        break

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Final Model Comparison Results\n",
        "\n",
        "Based on our training pipeline, the final metrics for each model are as follows:\n",
        "\n",
        "| Model | RMSE | MAE | MAPE (%) | R² |\n",
        "| :--- | :--- | :--- | :--- | :--- |\n",
        "| **LightGBM** | **45.39** | 21.41 | 16.08 | **0.858** |\n",
        "| Random Forest | 45.53 | **20.98** | **15.28** | 0.857 |\n",
        "| XGBoost | 45.98 | 21.71 | 16.36 | 0.854 |\n",
        "| CatBoost | 47.02 | 23.15 | 18.55 | 0.848 |\n",
        "| Linear Regression | 48.11 | 22.65 | 16.09 | 0.841 |\n",
        "\n",
        "**Conclusion**: LightGBM slightly outperforms the other models in terms of RMSE and R², making it the best candidate for our AQI predictions."
    ]
}

if target_idx is not None:
    notebook['cells'].insert(target_idx, markdown_cell)
else:
    notebook['cells'].append(markdown_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated successfully.")
