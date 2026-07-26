# SV-RCNet_BA_STO

Dataset configuration for surgical phase recognition in laparoscopic gastrectomy.

This repository contains NumPy arrays organized using a fixed 28:4:8 split for training, validation, and testing.

## Dataset Split

| Split | Cases | NumPy files |
|---|---:|---:|
| Training | 28 | 52 |
| Validation | 4 | 4 |
| Test | 8 | 8 |
| **Total** | **40** | **64** |

## Directory Structure

```text
dataset/
├── train/
│   ├── STOa_PS001_STOs01/
│   ├── ...
│   └── STOa_PS028_STOs01/
├── val/
│   ├── STOa_PS029_STOs01/
│   ├── ...
│   └── STOa_PS032_STOs01/
└── test/
    ├── STOa_PS033_STOs01/
    ├── ...
    └── STOa_PS040_STOs01/
```

## Included NumPy Arrays

All 40 cases contain a surgical phase annotation array:

```text
STOa_PSXXX_STOs01_phase.npy
```

Cases `PS022` through `PS027` additionally contain detector-derived metadata arrays:

```text
STOa_PSXXX_STOs01_gauze.npy
STOa_PSXXX_STOs01_tool.npy
STOa_PSXXX_STOs01_vessel.npy
STOa_PSXXX_STOs01_organ.npy
```

## Data Summary

- Number of surgical cases: 40
- Number of training cases: 28
- Number of validation cases: 4
- Number of test cases: 8
- Number of phase annotation arrays: 40
- Number of detector-derived metadata arrays: 24
- Total number of NumPy arrays: 64

## File Naming Convention

Each case is stored in a case-specific directory.

```text
STOa_PSXXX_STOs01/
```

The files inside each directory use the same case identifier.

```text
STOa_PSXXX_STOs01_phase.npy
STOa_PSXXX_STOs01_tool.npy
STOa_PSXXX_STOs01_organ.npy
STOa_PSXXX_STOs01_gauze.npy
STOa_PSXXX_STOs01_vessel.npy
```

## Validation

The repository structure is validated for:

- a 28:4:8 train/validation/test split
- 40 case directories
- sequential case identifiers from `PS001` to `PS040`
- consistency between directory names and NumPy filenames
- 64 NumPy files in total
- phase arrays for every case
- additional metadata arrays for cases `PS022`–`PS027`

## Repository Contents

```text
SV-RCNet_BA_STO/
├── dataset/
├── README.md
├── prepare_sto_dataset.py
├── publish_svrcnet_ba_sto.py
└── split_mapping.csv
```
