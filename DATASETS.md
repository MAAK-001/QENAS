# Datasets — discovered structure and resulting splits

The three dataset folders were inspected recursively before the loaders were written. They have **different**
layouts, so each has its own discovery function (`qenas/datasets/discovery.py`). Re-run the inspection at
any time:

```bash
python scripts/inspect_datasets.py --tree
```

Every experiment also stores the full discovery report (`dataset_report.json`) and the exact split
(`split.json`, list of sample IDs and file paths per split).

## BUSI — `Datasets/BUSI (Breast Ultrasound Image)/`

```
benign/     891 PNG  = 437 images + 454 masks   (16 images have 2-3 masks: "_mask", "_mask_1", "_mask_2")
malignant/  421 PNG  = 210 images + 211 masks   (1 image with 2 masks)
normal/     266 PNG  = 133 images + 133 masks   (all masks empty)
```

* Pairing: `<class> (<n>).png` ↔ `<class> (<n>)_mask[_k].png`; all images have ≥ 1 mask, no orphan masks.
* Images: RGB PNG (grey ultrasound stored as RGB), 544 distinct sizes (~ 500 × 450 to 780 × 600).
* Masks: mode "1" (binary) except 2 RGBA masks with values {0, 255}; multiple masks → union.
* No official split, no patient IDs.
* Exact duplicate: `benign (433)` is pixel-identical to `malignant (145)` but their masks differ
  (IoU 0.77) → both excluded (ambiguous ground truth).
* Used: benign + malignant (as in Mixed-GGNAS: 647 images) − 2 conflicting duplicates = **645**.

| split | images | benign / malignant |
|---|---:|---|
| Train | 413 | 279 / 134 |
| Validation | 103 | 70 / 33 |
| Test | 129 | 87 / 42 |

## CVC-ClinicDB — `Datasets/CVC-ClinicD (Polyp)/`

```
PNG/Original/      612 PNG  (384 × 288 RGB)
PNG/Ground Truth/  612 PNG  (RGB, identical channels, 0/255 with anti-aliasing grey values)
TIF/Original, TIF/Ground Truth   612 + 612 TIF (same frames; not decodable by Pillow → unused)
metadata.csv       frame_id → sequence_id (29 colonoscopy sequences)
class_dict.csv     background (0,0,0), polyp (255,255,255)
```

* Pairing: identical file names `<frame>.png`; all 612 matched; all frames present in metadata.
* Masks: up to 8.2 % intermediate grey pixels in a few masks (anti-aliasing) → binarised at > 127.
* No official split. Frames of a sequence are near-duplicates → **sequence-level split** (whole sequences
  per split).

| split | frames | sequences |
|---|---:|---:|
| Train | 395 | 18 |
| Validation | 96 | 5 |
| Test | 121 | 6 |

## IDRiD — `Datasets/IDRID/A.%20Segmentation/A. Segmentation/`

```
1. Original Images/a. Training Set/   54 JPG  IDRiD_01 … IDRiD_54   (4288 × 2848 RGB)
1. Original Images/b. Testing Set/    27 JPG  IDRiD_55 … IDRiD_81
2. All Segmentation Groundtruths/<split>/
     1. Microaneurysms/  IDRiD_xx_MA.tif   train 54, test 27
     2. Haemorrhages/    IDRiD_xx_HE.tif   train 53, test 27
     3. Hard Exudates/   IDRiD_xx_EX.tif   train 54, test 27
     4. Soft Exudates/   IDRiD_xx_SE.tif   train 26, test 14
     5. Optic Disc/      IDRiD_xx_OD.tif   train 54, test 27
```

(`B. Disease Grading` and `C. Localization` belong to other IDRiD sub-challenges and are not used.)

* Masks: palette TIFs with indices {0, 1}; a missing lesion file means the lesion is absent.
* Target: **optic disc** (default; complete for all 81 images). `--idrid-target MA|HE|EX|SE|LESIONS`
  selects a lesion or the union of the four lesion types.
* Official split kept; validation carved from the official training set. No patient IDs.

| split | images |
|---|---:|
| Train | 43 |
| Validation | 11 |
| Test (official) | 27 |
