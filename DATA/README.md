# DATA

This folder holds the project’s datasets. Large files are **not stored in the repo**. Instead, you download a single archive from Google Drive and the script places everything where the code expects it.

```
DATA/
  raw/                 # created by the download (ignored by git)
  data_manifest.csv    # tells the script where the archive is and how to check it
  get_data.py          # downloads, verifies, and unpacks the archive
```

---

## What’s in the data?

After download and unpack, you will have about **\~1.3 GB** under `DATA/raw/` with four groups:

1. **3k\_buses/** — 3,000-bus New England network

   * Capacity factor files for **wind** and **solar** (by year).
   * Network metadata (branches, HV buses, county mapping).

2. **17\_zones/** — 17-zone network

   * Static network CSVs (no time series).

3. **67\_counties/** — county-level dataset

   * Capacity factors for **wind** and **solar** (historical and future scenarios).
   * **Hourly demand** files (one CSV per year).
   * Network metadata (lines, nodes, county list).

4. **ne\_population/** — New England population and boundaries

   * Census boundary shapefiles (states and counties).
   * A derived `ne_population.csv`.
   * A small script and README describing the extraction.

> “Capacity factor” means the fraction of maximum possible output (0 to 1) for a generator technology.

---

## How to get the data

1. Open a terminal in the repository root.
2. Install the tiny helper once:

   ```bash
   pip install gdown
   ```
3. Run the downloader:

   ```bash
   python DATA/get_data.py
   ```

That’s it. The script will:

* Read **`DATA/data_manifest.csv`** to find one Google Drive file (`raw.zip`) plus its expected checksum and size.
* Download the archive as **`DATA/raw.zip`** (about **\~0.5 GB**).
* **Verify** the download is correct.
* **Unpack** everything into **`DATA/raw/`** and get it ready for the code.

---

## What are these two small files?

* **`get_data.py`**
  A tiny helper that downloads one zip from Google Drive, checks it is intact, and unpacks it into `DATA/raw/`.

* **`data_manifest.csv`**
  A one-line CSV that tells the script:

  * the Google Drive **file ID** of the archive,
  * the **checksum** to verify the download, and
  * the exact **file size**.

You do not edit these during normal use.

---
