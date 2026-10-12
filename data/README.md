# Data placement

This folder is intentionally empty of image data (too large to distribute with
the code). Download the public datasets below and place them exactly as shown;
`model/common.py` reads these paths.

## HAM10000

Source: Harvard Dataverse, https://doi.org/10.7910/DVN/DBW86T
(Tschandl P, Rosendahl C, Kittler H. *Sci Data* 2018;5:180161.)

```
data/HAM10000/HAM10000_metadata.csv
data/HAM10000/HAM10000_images_part_1/*.jpg
data/HAM10000/HAM10000_images_part_2/*.jpg
```

Also download the expert lesion segmentation masks from the same Dataverse
record (file `HAM10000_segmentations_lesion_tschandl.zip`) and extract so that:

```
data/HAM10000_masks/HAM10000_segmentations_lesion_tschandl/<image_id>_segmentation.png
```

exists for every image (10,015 masks).

## PAD-UFES-20

Source: Mendeley Data, https://doi.org/10.17632/zr7vgbcyr2.1
(Pacheco AGC et al. *Data Brief* 2020;32:106221.)

```
data/PAD-UFES-20/metadata.csv
data/PAD-UFES-20/images/*.png
```

No other data is required. Everything downstream (preprocessing caches,
trained weights, per-seed logits/saliency maps, figures, tables) is produced
by the scripts in `model/` from these two public sources.
