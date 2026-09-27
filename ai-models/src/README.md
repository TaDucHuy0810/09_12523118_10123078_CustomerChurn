# Model source entry points

The canonical pipeline remains in the repository-root `src/` directory to avoid duplicate implementations. These compatibility entry points satisfy the AI Models workflow layout:

- `python ai-models/src/preprocess.py`: load/clean data, split it, and preview the preprocessing transform fit on train only.
- `python ai-models/src/train.py`: run the canonical model training and artifact export.
- `python ai-models/src/evaluate.py`: regenerate metric and confusion-matrix figures.

The AI Models notebooks call the same root implementation, so schema, split, metrics, and serving artifacts stay aligned.
