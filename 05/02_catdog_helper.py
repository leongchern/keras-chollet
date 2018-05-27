import os, shutil

orig_dataset_dir = 'C:/Users/DP/Desktop/Projects/MCL/keras-chollet/05/catdog_orig'

base_dir = 'catdog_img'
os.makedirs(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(base_dir,'val')
os.mkdir(val_dir)
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(val_dir, 'cats')
os.mkdir(val_cats_dir)

val_dogs_dir = os.path.join(val_dir, 'dogs')
os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(train_cats_dir, f)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(val_cats_dir, f)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(test_cats_dir, f)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(train_dogs_dir, f)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(val_dogs_dir, f)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for f in fnames:
    src = os.path.join(orig_dataset_dir, f)
    dst = os.path.join(test_dogs_dir, f)
    shutil.copyfile(src, dst)

print('total train cat images:', len(os.listdir(train_cats_dir)))
print('total val cat images:', len(os.listdir(val_cats_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))

print('total train dog images:', len(os.listdir(train_dogs_dir)))
print('total val dog images:', len(os.listdir(val_dogs_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))