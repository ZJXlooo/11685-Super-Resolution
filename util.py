import shutil
from glob import glob
import zipfile
import urllib.request
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from random import random, sample
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from torchvision.utils import save_image

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()

def download_dataset():
    DIV2K_train_HR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    DIV2K_train_LR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
    DIV2K_valid_HR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    DIV2K_valid_LR = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not os.path.exists(os.path.join('train_hr')):
        os.makedirs(os.path.join('train_hr'))
    if not os.path.exists(os.path.join('train_lr')):
        os.makedirs(os.path.join('train_lr'))
    if not os.path.exists(os.path.join('valid_hr')):
        os.makedirs(os.path.join('valid_hr'))
    if not os.path.exists(os.path.join('valid_lr')):
        os.makedirs(os.path.join('valid_lr'))

    print('[!] Downloading Dataset')
    download_url(DIV2K_train_HR, os.path.join('temp', 'DIV2K_train_HR.zip'))
    download_url(DIV2K_train_LR, os.path.join('temp', 'DIV2K_train_LR_bicubic_X4.zip'))    
    download_url(DIV2K_valid_HR, os.path.join('temp', 'DIV2K_valid_HR.zip'))
    download_url(DIV2K_valid_LR, os.path.join('temp', 'DIV2K_valid_LR_bicubic_X4.zip'))

    print('[!] Upzip zipfile')
    unzip_zip_file(os.path.join('temp','DIV2K_train_HR.zip'), 'temp')
    unzip_zip_file(os.path.join('temp','DIV2K_train_LR_bicubic_X4.zip'), 'temp')
    unzip_zip_file(os.path.join('temp','DIV2K_valid_HR.zip'), 'temp')
    unzip_zip_file(os.path.join('temp','DIV2K_valid_LR_bicubic_X4.zip'), 'temp')

    print('[!] Reformat DIV2K HR (Training Set)')
    image_path = glob('temp/DIV2K_train_HR/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('train_hr', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR (Training Set)')
    image_path = glob('temp/DIV2K_train_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('train_lr', f'{index:04d}.png'))

    print('[!] Reformat DIV2K HR (Validation Set)')
    image_path = glob('temp/DIV2K_valid_HR/*.png')
    # print(image_path)
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('valid_hr', f'{index:04d}.png'))

    print('[!] Reformat DIV2K LR (Validation Set)')
    image_path = glob('temp/DIV2K_valid_LR_bicubic/X4/*.png')
    image_path.sort()
    for index, path in enumerate(image_path):
        shutil.move(path, os.path.join('valid_lr', f'{index:04d}.png'))

    shutil.rmtree('temp')

def crop_image(hr_path, lr_path, hr_size, lr_size):
    # image_size: crop之后的图像长宽
    image_list = os.listdir(hr_path)

    for index, image_name in enumerate(image_list):
        if not image_name.endswith('.png'):
            continue
        hr_img = cv2.imread(os.path.join(hr_path, image_name))
        lr_img = cv2.imread(os.path.join(lr_path, image_name))
        height, width, channels = hr_img.shape
        num_row = height // hr_size
        num_col = width // hr_size
        image_index = 0

        if index % 500 == 0:
            print(f'[*] [{index}/{len(image_list)}] Make patch {os.path.join(hr_path, image_name)}')
        
        num_row = sample([i for i in range(num_row)], 4)
        num_col = sample([i for i in range(num_col)], 4)

        for i in num_row:
            if (i+1)*hr_size > height:
                break
            for j in num_col:
                if (j+1)*hr_size > width:
                    break
                cv2.imwrite(os.path.join(hr_path, f'{image_name.split(".")[0]}_{image_index}.png'),
                            hr_img[i*hr_size:(i+1)*hr_size, j*hr_size:(j+1)*hr_size])
                cv2.imwrite(os.path.join(lr_path, f'{image_name.split(".")[0]}_{image_index}.png'),
                            lr_img[i*lr_size:(i+1)*lr_size, j*lr_size:(j+1)*lr_size])
                image_index += 1
        os.remove(os.path.join(hr_path, image_name))
        os.remove(os.path.join(lr_path, image_name))

def resize_image(path, image_size, change_path=None):
    # image_size: resize之后的图像长宽
    image_list = os.listdir(path)

    for index, image_name in enumerate(image_list):
        if not image_name.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(path, image_name))
        # print(img.size)
        img = cv2.resize(img, (image_size, image_size))

        if index % 500 == 0:
            print(f'[*] [{index}/{len(image_list)}] Make patch {os.path.join(path, image_name)}')

        if not change_path:
            cv2.imwrite(os.path.join(path, image_name), img)
        else:
            cv2.imwrite(os.path.join(change_path, image_name), img)

# 定义dataloader和dataset

class Datasets(Dataset):
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'train' or self.mode == 'valid':
          self.image_file_name = sorted(os.listdir(os.path.join(self.mode+'_lr')))
        else:
          self.image_file_name = sorted(os.listdir(os.path.join(self.mode)))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        if self.mode == 'train' or self.mode == 'valid':
          high_resolution = Image.open(os.path.join(self.mode+'_hr', file_name)).convert('RGB')
          low_resolution = Image.open(os.path.join(self.mode+'_lr', file_name)).convert('RGB')

          if random() > 0.5:
              high_resolution = TF.vflip(high_resolution)
              low_resolution = TF.vflip(low_resolution)

          if random() > 0.5:
              high_resolution = TF.hflip(high_resolution)
              low_resolution = TF.hflip(low_resolution)

          high_resolution = TF.to_tensor(high_resolution)
          low_resolution = TF.to_tensor(low_resolution)  
        else:
          high_resolution = file_name
          low_resolution = Image.open(os.path.join(self.mode, file_name)).convert('RGB')
          low_resolution = TF.to_tensor(low_resolution)  

        images = {'lr': low_resolution, 'hr': high_resolution}
        return images

    def __len__(self):
        return len(self.image_file_name)

def get_test_images(test_label_path, test_result_path, restore_path):
    print('In get test images')
    
    data = {}
    for label, prediction in zip(sorted(os.listdir(test_label_path)), sorted(os.listdir(test_result_path))):
        if label != prediction:
            print('Error: filenames dont match')
            break
        else:
            #load label
            filename = label
            
            image_label = Image.open(test_label_path + filename)
            x = TF.to_tensor(image_label).numpy()
            
            image_prediction = Image.open(test_result_path + filename)
            y = TF.to_tensor(image_prediction).numpy()
            
            
            if x.shape != y.shape:
                image_prediction = transforms.Resize(x.shape[1:])(image_prediction)
                y = TF.to_tensor(image_prediction).numpy()
                save_image(TF.to_tensor(y.transpose(1,2,0)), restore_path+filename)

                
            data[filename] = (x.transpose(1,2,0), y.transpose(1,2,0))
    # print(len(data))
    return data

def compute_scores(test_label_path, test_result_path, restore_path=None):
    print('In computer scores')
    data = get_test_images(test_label_path, test_result_path, restore_path)

    
    total_images = len(data)
    psnr_val = 0
    ssim_val = 0
    
    for filename in data:
        label, prediction = data[filename]
        if label.shape[2] == 1:
          psnr_val += peak_signal_noise_ratio(label[:, :, 0], prediction.mean(axis=2))
          ssim_val += structural_similarity(label[:, :, 0], prediction.mean(axis=2))
        else:
          psnr_val += peak_signal_noise_ratio(label, prediction)
          ssim_val += structural_similarity(label, prediction, multichannel=True)
        
    psnr_val /= total_images
    ssim_val /= total_images
    
    print(psnr_val, ssim_val)