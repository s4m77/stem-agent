# DSBench Inspection

This report is exploratory. DABench remains the validated StemDS headline benchmark.

## Summary
- Root: `external/DSBench`
- Exists: `True`
- Data-analysis metadata records: `38`
- Data-modeling metadata records: `74`
- Simple convertible records: `0`
- Conversion supported now: `False`

## Relevant Directories And Files
### Directory structure
- `LICENSE`
- `README.md`
- `data_analysis/`
- `data_modeling/`
- `figures/`
- `requirments.txt`
### Metadata files
- `data_analysis/data.json`
- `data_modeling/data.json`
### Task files
- None detected.
### Data files
- `data_analysis/eva_autogen_gpt.ipynb`
- `data_analysis/eval_Claude.ipynb`
- `data_analysis/eval_GPT.ipynb`
- `data_analysis/eval_Gemini.ipynb`
- `data_analysis/eval_code_intepreter_gpt4o.ipynb`
- `data_modeling/eva_autogen_gpt.ipynb`
- `data_modeling/eval_code_interpreter.ipynb`
### Zip files
- `data_analysis/data_old.zip`
- `data_modeling/output_model.zip`
- `data_modeling/save_performance.zip`
### Zip sample members
- `external/DSBench/data_analysis/data_old.zip::data/.DS_Store`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/MO13-Data-Analysis-Data-Workbook.xlsx`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question5.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question4.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question16.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question14.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question6.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question7.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question15.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question11.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question3.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/.DS_Store`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question2.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question10.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question12.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question1.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question13.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/introduction.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question9.txt`
- `external/DSBench/data_analysis/data_old.zip::data/00000034/question8.txt`

## Detected Modalities
- `code`
- `excel`
- `image`
- `notebook`
- `tabular_text`
- `text_questions`

## Example Records
### Data Analysis
```json
{
  "answers": [
    "D",
    "D",
    "I",
    "A",
    "H",
    "C",
    "E",
    "H",
    "D",
    "A",
    "I",
    1661626,
    323272
  ],
  "id": "00000001",
  "name": "2016-round-1-section-2-chip-off-the-old-block",
  "questions": [
    "question6",
    "question7",
    "question8",
    "question9",
    "question10",
    "question11",
    "question12",
    "question13",
    "question14",
    "question15",
    "question16",
    "question17",
    "question18"
  ],
  "txt": "",
  "url": "https://www.eloquens.com/tool/vYDF7b/finance/modeloff-sample-past-questions/2016-round-1-section-2-chip-off-the-old-block",
  "year": 2016
}
```

### Data Modeling
```json
{
  "name": "titanic",
  "size": "93.08 kB",
  "url": "https://www.kaggle.com/competitions/titanic",
  "year": 2012
}
```

### Simple Convertible
```json
null
```

## Example Data File
- `external/DSBench/data_analysis/data_old.zip::data/00000034/MO13-Data-Analysis-Data-Workbook.xlsx`

## Recommended Conversion Strategy
Treat DSBench as a future multi-modal/multi-file benchmark extension. First add a dedicated extractor for archived question files and Excel workbooks, then define a conservative subset that can be scored with StemDS metrics.

## Blockers And Unknowns
- Native data-analysis metadata lists question ids and answers, but question text and data files live separately in archives with mixed file types.
- Detected Excel/image assets; current DataAnalysisTask supports CSV-style single-table tasks only.
- Data-modeling metadata lists competitions, but local train/test files, target columns, and metrics are not explicit in the top-level metadata.
- No explicit simple CSV/question/answer subset metadata was found.

## Unknowns
- How to map ModelOff-style multiple-choice/numeric answers to StemDS scalar answer contracts.
- Whether all required analysis assets can be read without Excel/image-specific tooling.
- Which DSBench modeling metric and target field should map to each MLEngineeringTask.
