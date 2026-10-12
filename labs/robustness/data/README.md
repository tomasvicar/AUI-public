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

## `heart_disease_hungarian.csv`, `heart_disease_switzerland.csv`, `heart_disease_va_long_beach.csv`

The other three hospitals of the same UCI data set: Hungarian Institute of
Cardiology, Budapest (294 patients); University Hospitals of Zürich and Basel
(123); VA Medical Center, Long Beach (200). Same 13 features and target `num`
as the Cleveland file. The lecture uses them as "a different hospital" on our
own model (decision of the author, 12 October 2026).

- Source: the `processed.hungarian.data`, `processed.switzerland.data` and
  `processed.va.data` files of the UCI archive
  <https://archive.ics.uci.edu/static/public/45/heart+disease.zip>, same
  authors and DOI as above. Licence: **CC BY 4.0**.
- Converted on 12 October 2026 to CSV with the column names of the
  Cleveland file, values unchanged (`?` became an empty field). The
  `processed.cleveland.data` of the same archive converts to exactly the
  Cleveland file above.
- Missing values are many (`ca` empty for 96-99 % of the patients, `thal` for
  42-90 %). A cholesterol of **0** (all 123 Swiss patients, 49 at VA) and a
  resting blood pressure of 0 (one VA patient) mean "not measured"; the
  file keeps them as 0, `heart.load_site_raw` turns them into missing values.

## PneumoniaMNIST

Not stored here. The notebook downloads `pneumoniamnist.npz` (4.2 MB, 28 x 28
chest X-rays, 4708 / 524 / 624 train / validation / test images, labels
normal / pneumonia) from the MedMNIST record on Zenodo,
<https://zenodo.org/records/10519652>, licence CC BY 4.0. Yang et al. (2023),
*MedMNIST v2*, Scientific Data, doi:10.1038/s41597-022-01721-8.
