import cv2
import numpy as np
import os
import subprocess


def apply_blur(image, kernel_size, sigma):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image
def apply_fast_fading(compressed_image, snr):
    noise_std = np.std(compressed_image) / (10**(snr/20))
        
    noise_real = np.random.randn(*compressed_image.shape)
    noise_imaginary = np.random.randn(*compressed_image.shape)
    noise = noise_std * (noise_real + 1j * noise_imaginary)

    distorted_image = compressed_image + noise
    mag = np.abs(distorted_image)

    
    lp = cv2.normalize(mag, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    lp = np.clip(lp, 0, 1)
    lp = (255*lp).astype(np.uint8)
    return lp

def apply_jpeg_compression(image, quality):
    _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    decoded_image = cv2.imdecode(encoded_image, 1)
    return decoded_image

def apply_jpeg2000_compression(image_name, image_path, bitrate, i):

    output_image_path = jp2k_folder+image_name+"_"+str(i)+".bmp"
    subprocess.run(["kdu_compress", "-i", image_path, "-o", output_image_path, "-rate", f"{bitrate}L"])


def apply_white_gaussian_noise(image, var):
    sigma = np.sqrt(var)
    noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


folder_path = "path-to-pristine-images"
blur_folder = "create-path-for-blur"
ff_folder = "create-path-for-ff"
jpeg_folder = "create-path-for-jpeg"
jp2k_folder = "create-path-for-jp2k"
wn_folder = "create-path-for-wn"


# h = 0
for filename in os.listdir(folder_path):

    image_name = filename[:-4]

    image = cv2.imread(folder_path+filename)


    #applying blur:
    i = 1
    kernel_size = 15
    for sigma in np.arange(0.0, 30, 1):
        save_image_name = blur_folder+image_name+"_"+str(i)+".bmp"
        blurred_image = apply_blur(image, kernel_size, sigma)
        cv2.imwrite(save_image_name, blurred_image)  
        i+=1

    # applying ff:
    i = 1
    for snr in range(1, 30, 1):
        save_image_name = ff_folder+image_name+"_"+str(i)+".bmp"
        faded_image = apply_fast_fading(image, snr)
        cv2.imwrite(save_image_name, faded_image)  
        i+=1

    # applying jpeg:
    i = 1
    for jpeg_quality in range(3, 50, 4):
        save_image_name = jpeg_folder+image_name+"_"+str(i)+".bmp"
        # jpeg_compressed_image = apply_jpeg_compression(image, jpeg_quality)
        jpeg_compressed_image = apply_jpeg_compression(image, jpeg_quality)
        cv2.imwrite(save_image_name, jpeg_compressed_image)  
        i+=1

    # applying jp2k:
    i = 1
    for bitrate in np.arange(0.04, 3.16, 0.1):
        save_image_name = jp2k_folder+image_name+"_"+str(i)+".bmp"
        tiff_name = folder_path+image_name+".tiff"
        cv2.imwrite(tiff_name, image)
        jpeg2000_compressed_image = apply_jpeg2000_compression(image_name, tiff_name, bitrate, i)
        i+=1



    # applying wn:
    i = 1
    for sigma in np.arange(0.001, 1.2, 0.1):
        save_image_name = wn_folder+image_name+"_"+str(i)+".bmp"
        noisy_image = apply_white_gaussian_noise(image, sigma)
        cv2.imwrite(save_image_name, noisy_image)  
        i+=1
