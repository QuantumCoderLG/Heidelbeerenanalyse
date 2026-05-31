# Blueberry Quality Dataset

Version: `v1.0.0`

This dataset supports blueberry instance segmentation and multi-stage quality
classification. It is published separately from the source-code repository.

## Archive contents

`blueberry-source-images-v1.0.0.zip` contains:

- `data/raw/`: segmentation source images and XML annotations.
- `data/all_images/Ampel/`: labeled quality-class images.
- `data/all_images/Heidelbeeren2/`: additional blueberry images.
- `data/BBoxes_annotation_data/`: additional bounding-box annotations.

`blueberry-curated-crops-v1.0.0.zip` contains:

- `data/instance_crops/images/`: curated single-instance crops.
- `data/instance_crops/masks/`: binary crop masks.
- `data/instance_crops/instance_masks/`: instance masks.
- `data/instance_crops/metadata/`: metadata tables.
- `data/instance_crops/splits/`: manual train, validation and test splits.
- `data/instance_crops/rejections/`: rejected crop records.

Generated overlays, processed exports and duplicate `_split` images are
excluded.

## Image metadata

Original image metadata including EXIF is intentionally retained. Users must
review metadata before redistribution if their use case has different privacy
requirements.

## License and citation

Dataset license: `CC BY 4.0`

Zenodo DOI: `10.5281/zenodo.20479053`

Source code citation metadata lives in `CITATION.cff` in the GitHub repository.
