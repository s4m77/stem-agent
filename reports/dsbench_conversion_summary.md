# DSBench Subset Conversion Summary

This conversion is exploratory. DABench remains the validated StemDS benchmark result.

- Root: `tests/fixtures/fake_dsbench`
- Output: `/private/var/folders/dt/4wkthmmj16130yh3rzlsq_p80000gn/T/pytest-of-s4m/pytest-24/test_convert_dsbench_subset_cl0/dsbench_subset.jsonl`
- Converted tasks: `1`
- Skipped records: `0`

## Converted Tasks
- `dsbench_sales_total` -> `tests/fixtures/fake_dsbench/tables/sales.csv`

## Skipped Records
- None.

## Next Steps
- Add explicit DSBench subset metadata for CSV/table tasks if a conservative analysis subset is desired.
- Add separate extractors before supporting Excel, image, notebook, or Kaggle-style modeling tasks.
