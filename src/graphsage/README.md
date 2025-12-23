# GraphSAGE Pipeline - Quick Start

## Structure

```
src/graphsage/
├── data.py      # Data loading (CSV → PyTorch Geometric)
├── model.py     # GraphSAGE model
├── train.py     # Training & evaluation
└── main.py      # Run pipeline
```

## Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd src/graphsage
python3 main.py
```

## What You'll See

**Progress bars** for:
1. **Data loading**: Processing 5M+ transactions
   ```
   Processing transactions: 100%|████████| 5078345/5078345 [02:15<00:00, 37.5k it/s]
   ```

2. **Training**: Live metrics every 10 epochs
   ```
   Training: 45%|████▌     | 45/100 [00:30<00:37, Loss: 0.1234, Train F1: 0.8567, Val F1: 0.8234]
   ```

## Output

- **Model**: Saved to `outputs/graphsage_model.pt`
- **Metrics**: Precision, Recall, F1, Accuracy on train/val/test sets

## Next Steps

After training, you can:
1. Load the model for predictions
2. Add GNNExplainer for xAI
3. Compare with HGMAE (Phase 3)
