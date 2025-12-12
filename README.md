# Embedded HW4 — EE 4065

## Homework scope

- Q1 (Section 10.9): Handwritten Digit Recognition from Digital Images.
- Q2 (Section 11.8): Handwritten Digit Recognition from Digital Images (second pipeline).
- Deliverables: code, trained artifacts, English LaTeX report, GitHub repository, email to instructor.

## Repo layout

- `src/`: training/inference scripts (placeholders ready for dataset integration).
- `data/`: offline datasets (not included; place required files here).
- `models/`: trained models/checkpoints/converted binaries.
- `report/`: LaTeX report (`main.tex`) and build outputs.

## Requirements & tooling

- Python 3.10+ with numpy (local venv recommended).
- LaTeX: `/Library/TeX/texbin/pdflatex` (see `Makefile`).
- No network access in this environment; datasets must be provided locally.

## Suggested workflow

1. Place the offline datasets in `data/` as referenced by the book sections.
2. (Done) `src/train_q1.py` and `src/train_q2.py` implement MNIST pipelines using numpy; they expect `data/MNIST-dataset` with IDX files.
3. (Optional) Create venv and install numpy:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install numpy`
4. Train:
   - `source .venv/bin/activate`
   - `python src/train_q1.py`
   - `python src/train_q2.py`
5. Store trained outputs in `models/` and sample predictions/figures in `report/figures/`.
6. Update `report/main.tex` with methodology, experiments, results, and deployment details.
7. Build the report: `make -C report`.
8. Push to GitHub and send the repository link via email as required.

## Notes

- Replace placeholder TODOs in code/report with actual results once datasets and target MCU details are available.
- If targeting STM32, consider TFLite Micro or STM32Cube.AI for deployment; document memory/latency metrics.
