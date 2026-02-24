# Data Directory Policy

`data/` is reserved for real experimental microscopy/FRAP datasets and is intentionally ignored by git.

## What belongs here

- Raw TIFF/ND2/LIF/CZI image stacks
- Lab-internal spreadsheets and metadata exports
- Large intermediate files generated during analysis

## What should be committed instead

- Small synthetic/openly licensed examples in `sample_data/`
- Documentation of expected outputs and test fixtures

## Getting test data

Use `sample_data/` for reproducible tests. If your workflow requires larger proprietary datasets, place them in this folder locally.
