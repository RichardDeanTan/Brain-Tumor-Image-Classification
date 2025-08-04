import os
import shutil

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
SPLITS = ['Train', 'Val']
DST_ROOT = 'Dataset'
IMG_EXTS = ('.jpg', '.jpeg', '.png')

for split in SPLITS:
    for cls in CLASSES:
        src_dir = os.path.join(split, cls, 'images')
        dst_dir = os.path.join(DST_ROOT, split, cls)

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                if fname.lower().endswith(IMG_EXTS):
                    shutil.copy2(
                        os.path.join(src_dir, fname),
                        os.path.join(dst_dir, fname)
                    )