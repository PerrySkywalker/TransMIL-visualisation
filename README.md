# TransMIL-visualisation

Supports up to **200,000 features** per Whole Slide Image (WSI) with **24GB of GPU memory**.

**The feature extractor for the experiment is Ctranspath, the aggregator is TransMIL, and the dataset is sourced from camelyon16.**

# First

After using extract_features_fp.py in CLAM you will get .h5 files, put these in the h5-files folder. Put the WSI thumbnails in the images folder. Put the xxx.ckpt to model folder.

Modify the source code of nystrom_attention, the modified code is in nystrom_attention.py

# Second

run main.py. You need to pay attention to the parameters in main.py.

downsample

The maximum size of the WSI is 10000x10000, the size of the thumbnail is 100x100, then the downsample is 10000/100=100.

patch_size

Your patch_size at 20x magnification is 224x224, so here it should be 448x448

You can adjust the min_threshold and max_threshold to make the heatmap look better.



If you have any questions, let me know.


