# Original data

Download the original dataset from
[Zenodo](https://doi.org/10.5281/zenodo.17954508).
If you want to rebuild the processed data, download all regional ZIPs and put them
directly in `data/raw_data/`, without extracting them. Then run from the project root:

```sh
python data/prepare_dataset.py
```

The script creates `data/processed_data/` and organizes the rasters into site
folders and tiles. Existing completed regions are checked and reused.
