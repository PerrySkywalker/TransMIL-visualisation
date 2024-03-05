# TransMIL-visualisation

I will upload my code by 12 March 2024 .


# First

After using extract_features_fp.py in CLAM you will get .h5 files, put these in the h5-files folder. Put the WSI thumbnails in the images folder. Put the xxx.ckpt to model folder.

Modify the source code of nystrom_attention, the modified code is in nystrom_attention.py


# Second
run main.py. You need to pay attention to the parameters in main.py.

downsample

The maximum size of the WSI is 10000x10000, the size of the thumbnail is 100x100, then the downsample is 10000/100=100.

patch_size

Your patch_size at 20x magnification is 224x224, so here it should be 448x448




If you have any questions, let me know.