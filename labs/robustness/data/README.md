# Data of the robustness lab

## `heart_disease_cleveland.csv`

The Cleveland subset of the **UCI Heart Disease** data set: 303 patients,
13 clinical features and the target `num` (0 = no disease, 1-4 = disease;
the lab uses `num > 0`). Missing values are kept as in the source (`ca`: 4,
`thal`: 2) and filled with the column median in the code.

- Source: Janosi, Steinbrunn, Pfisterer, Detrano (1988), *Heart Disease*, UCI
  Machine Learning Repository, <https://archive.ics.uci.edu/dataset/45/heart+disease>,
  doi:10.24432/C52P4X.
- Licence: **CC BY 4.0**.
- Exported on 24 September 2026 with `ucimlrepo.fetch_ucirepo(id=45)`
  (features plus the target column `num`), unchanged. It is vendored so that
  the notebooks do not depend on the UCI API being up during a lab.

The same file is the running example of the labs of *Explainability* and
*Bias and fairness*.

## PneumoniaMNIST

Not stored here. The notebook downloads `pneumoniamnist.npz` (4.2 MB, 28 x 28
chest X-rays, 4708 / 524 / 624 train / validation / test images, labels
normal / pneumonia) from the MedMNIST record on Zenodo,
<https://zenodo.org/records/10519652>, licence CC BY 4.0. Yang et al. (2023),
*MedMNIST v2*, Scientific Data, doi:10.1038/s41597-022-01721-8.
