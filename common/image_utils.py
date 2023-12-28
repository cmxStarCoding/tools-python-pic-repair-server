import argparse
import base64
import os
import random
import string
import hashlib
from cachetools import LRUCache

global_cache = {}

cache = LRUCache(maxsize=1000)


def cache_data(key, value):
    cache[key] = value


def get_cached_data(key):
    return cache.get(key)


def assemble_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))

    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=200, help='Tile size, 0 for no tile during testing')

    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', default='--fp32',
        help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()
    return args


def generate_random_string(length):
    # 所有的字符集
    chars = string.ascii_letters + string.digits
    # 使用random.choices随机选择4个字符
    random_string = ''.join(random.choices(chars, k=length))
    return random_string


def encrypt(text):
    encrypted_bytes = base64.b64encode(text.encode())
    return encrypted_bytes.decode()


def decrypt(encrypted_text):
    decrypted_bytes = base64.b64decode(encrypted_text.encode())
    return decrypted_bytes.decode()


def save_uploaded_file(file, upload_dir):
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
        f.close()
    return file_path


def get_file_extension(file_name):
    # 使用os.path.splitext来获取文件名和扩展名的元组
    _, extension = os.path.splitext(file_name)
    return extension


def md5_hash(text):
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    return md5.hexdigest()
