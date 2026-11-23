# Report template - diabetic-retinopathy referral

Write the report for **the board of the screening programme**: clinicians who
will decide whether your software may triage their patients. At most **four
pages** (about 2000 words) plus figures and tables, in the GitHub repository next to the code
(`report.md`, `report.pdf` or `report.html`, in the root). The structure follows the
[CLAIM checklist](https://doi.org/10.1148/ryai.240300) (Checklist for AI in
Medical Imaging), shortened for the project; the lecture *Validation, clinical
implementation and regulation* explains why each part is there.

Every number in the report comes from code in the repository. Give a 95 %
interval wherever you give a proportion (sensitivity, specificity, PPV).

## 1. Intended use

- What decision does the software make, for whom, and who acts on it?
- What happens to an image the software cannot judge?
- Where must it **not** be used?

## 2. Data

- What you received, per clinic: images, patients, label formats.
- What you found in the data and what you did about it. For each problem:
  how you found it, how many images it touches, what you decided and why.
- The flow from the export to the training set: how many images and patients
  were excluded at each step, and why (a small table or a diagram); every
  image with its fate is in `data_table.csv`.
- How you defined the reference standard (the label you train and evaluate
  against).

## 3. Method

- How you split the data and why that split answers the question the board asks.
- The model and the training in a few sentences; enough to rerun it.
  The settings you finally used come from the search in section 4.
- How you chose the operating point (the referral threshold), on which data,
  and **before** looking at your final test numbers.

## 4. Hyperparameter optimization

- The search space, the method and the budget; on which data the objective
  was measured.
- The search history in one figure.
- The best result against your default settings, and how far that gain can
  be trusted.

## 5. Results

- Sensitivity, specificity and PPV at your operating point, with 95 % intervals,
  per image and per patient.
- What the PPV and the number of referrals would be in a screening
  population, where most patients have no referable retinopathy. State the
  prevalence you assume and where it comes from.
- What you expect on a new batch (`expected.csv`) and why those intervals:
  which of your numbers you trust, and which ones the export may flatter.
- Performance per clinic, per camera and per patient group (age, sex) where the
  data allow it. Say when a group is too small to conclude anything.
- A few failures: what kind of images the software gets wrong.

## 6. Limitations and next steps

- What your results do **not** show.
- What would be needed before the software sees real patients.

## 7. Reproducibility and use of AI tools

- The commands that rebuild every number in the report, from the export to the
  predictions, and the one that reruns the hyperparameter search.
- Which AI tools you used and for what (code, text, figures). Using them is
  allowed; not declaring them is not.
