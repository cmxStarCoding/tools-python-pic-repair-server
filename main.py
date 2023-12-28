import time
import requests
from fastapi import FastAPI, Request, BackgroundTasks,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from basicsr.archs.rrdbnet_arch import RRDBNet
from common.image_utils import *
from real_esrgan.realesrgan.utils import RealESRGANer
from common.file_cache import FileCache
import os
import cv2
import glob
import imghdr
import asyncio
import configparser

# 创建一个ConfigParser对象
config = configparser.ConfigParser()

# 读取配置文件
config.read('config.ini')

app = FastAPI()
# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/image-plus/static", StaticFiles(directory="./static"), name="static")
host = config['uvicorn']['host']
port = int(config['uvicorn']['port'])
domain = config['app']['domain']
file_cache = FileCache()
# 加载模型公共参数
args = assemble_common_args()
# 加载模型
args.model_name = args.model_name.split('.')[0]
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
# file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
model_path = os.path.join('real_esrgan/weights', args.model_name + '.pth')
# if not os.path.isfile(model_path):
#     ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#     for url in file_url:
#         # model_path will be updated
#         model_path = load_file_from_url(
#             url=url, model_dir=os.path.join(ROOT_DIR, '../Real-ESRGAN/weights'), progress=True, file_name=None)

dni_weight = None
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=args.tile,
    tile_pad=args.tile_pad,
    pre_pad=args.pre_pad,
    half=not args.fp32,
    gpu_id=args.gpu_id
)
face_enhancer = None
# if args.face_enhance:  # Use GFPGAN for face enhancement
#     from gfpgan import GFPGANer
#
#     face_enhancer = GFPGANer(
#         model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
#         upscale=args.outscale,
#         arch='clean',
#         channel_multiplier=2,
#         bg_upsampler=upsampler)


async def long_running_task(paths, args):
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Starting', imgname)
        file_key = imgname + extension

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            # if args.face_enhance:
            #     _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False,
            #                                          paste_back=True)
            # else:
            output, _ = upsampler.enhance(img, outscale=args.outscale, file_key=file_key)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)
            os.remove(path)
            print('Ended', idx, imgname)


def run_in_thread(fn):
    def run(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(fn(*args, **kwargs))

    return run


@app.post("/image-plus/plus")
async def forward(request: Request, background_tasks: BackgroundTasks):
    data = await request.form()
    # name = data.get('name')
    # print(request.query_params.get('image'), '图片地址')
    video_list = f'img/save/'
    os.makedirs(video_list, exist_ok=True)
    outscale = float(data["upscaling_resize"])
    args.outscale = outscale
    scale_tuple = (2.0, 3.0, 4.0)
    if float(outscale) not in scale_tuple:
        outscale = 2.0
    args.outscale = outscale
    args.output = 'static/'
    os.makedirs(args.output, exist_ok=True)

    if "file" in data:
        print('上传文件逻辑')
        uploaded_file = data["file"]
        ext = get_file_extension(uploaded_file.filename)

        uploaded_file.filename = str(int(time.time())) + generate_random_string(4) + ext
        filename = uploaded_file.filename
        args.input = os.path.dirname(os.path.abspath(__file__)) + '/' + save_uploaded_file(uploaded_file, video_list)
    else:
        print('取图片链接逻辑')
        img_url = data["image"]
        # filename = 时间戳加随机字符串
        response = requests.get(img_url)
        args.input = ''
        max_size = 15 * 1024 * 1024
        if response.status_code == 200:
            if 'content-length' in response.headers and int(response.headers['content-length']) > max_size:
                raise HTTPException(status_code=422, detail={"code": "422", "message": "图片大小超过限制大小15M"})
            content_type = response.headers.get('content-type')
            ext = '.jpg'
            if 'image' in content_type:
                ext = '.'+imghdr.what(None, h=response.content)
            filename = str(int(time.time())) + generate_random_string(4) + ext
            # 保存图片到infile目录
            with open(f"{video_list}{filename}", "wb") as f:
                f.write(response.content)
                f.close()
            args.input = os.path.dirname(os.path.abspath(__file__)) + '/' + video_list + filename
        else:
            return {"task_id": ''}
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    send_email_thread = run_in_thread(long_running_task)
    background_tasks.add_task(send_email_thread, paths, args)
    task_id = md5_hash(filename)
    file_cache.set(task_id, filename)
    return {"task_id": task_id}


@app.get("/image-plus/plus/info")
async def plus_info(request: Request):
    # print(request.query_params.get('img_url'), '图片地址')
    task_id = request.query_params.get('task_id')
    fid = file_cache.get(task_id)
    if fid is None:
        return {'image_url': '', 'schedule': 0.00}

    # fid = fid.decode('utf-8')
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/static/' + fid
    print(file_path)
    # if port:
    #     image_url = host + ':' + str(port) + '/static/' + fid
    # else:
    image_url = domain + '/image-plus/static/' + fid

    # 使用os.path.exists检查文件是否存在
    if os.path.exists(file_path):
        return {'image_url': image_url, 'schedule': 1}
    else:
        hash_key = md5_hash(fid+'schedule')
        schedule = file_cache.get(hash_key)
        if schedule is None:
            schedule = 0.00

        # 文件比较大的情况下存在进度为1, 但文件尚未写入成功的情况(文件大写入比较慢)。此时也需要判断下文件是否存在
        if not os.path.exists(file_path) and float(schedule) == 1.00:
            return {'image_url': '', 'schedule': 0.95}

        return {'image_url': '', 'schedule': schedule}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app='main:app', host=host, port=port, use_colors=True, workers=4)
