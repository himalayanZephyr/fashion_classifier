from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from build_model import create_model
from gradcam import *
import cv2
import numpy as np

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_name = 'export.pkl'


pytorch_labels = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17, '26': 18, '27': 19, '28': 20, '29': 21, '3': 22, '30': 23, '31': 24, '32': 25, '33': 26, '34': 27, '35': 28, '36': 29, '37': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35, '43': 36, '44': 37, '46': 38, '47': 39, '48': 40, '5': 41, '6': 42, '7': 43, '8': 44, '9': 45}


reverse_map = {}
for k,v in pytorch_labels.items():
    reverse_map[v] = int(k) 
actual_names = ['None',
 'Anorak',
 'Blazer',
 'Blouse',
 'Bomber',
 'Button-Down',
 'Cardigan',
 'Flannel',
 'Halter',
 'Henley',
 'Hoodie',
 'Jacket',
 'Jersey',
 'Parka',
 'Peacoat',
 'Poncho',
 'Sweater',
 'Tank',
 'Tee',
 'Top',
 'Turtleneck',
 'Capris',
 'Chinos',
 'Culottes',
 'Cutoffs',
 'Gauchos',
 'Jeans',
 'Jeggings',
 'Jodhpurs',
 'Joggers',
 'Leggings',
 'Sarong',
 'Shorts',
 'Skirt',
 'Sweatpants',
 'Sweatshorts',
 'Trunks',
 'Caftan',
 'Cape',
 'Coat',
 'Coverup',
 'Dress',
 'Jumpsuit',
 'Kaftan',
 'Kimono',
 'Nightdress',
 'Onesie',
 'Robe',
 'Romper',
 'Shirtdress',
 'Sundress']

device = torch.device("cpu")#("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

path = Path(__file__).parent
print(path)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

async def load_model():
    try:
        model1 = create_model("supervised",path)
        model1.to(device)
        model2 = create_model("rotation",path)
        model2.to(device)
        return [model1,model2]
    except RuntimeError as e:
        raise(e)

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(load_model())]
models = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    #img = BytesIO(img_bytes)
    
    use_cuda = False #True

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    target_index = None

    ############# SUPERVISED MODEL ###############
    grad_cam = GradCam(model = models[0], target_layer_names = ["11"], use_cuda=use_cuda, arch="supervised")
    mask,prediction1 = grad_cam(input, target_index)
    show_cam_on_image(img, mask, path, "supervised")

    ############ UNSUPERVISED ROTATION MODEL ############
    grad_cam2 = GradCam(model = models[1], target_layer_names = ["6"], use_cuda=use_cuda, arch="rotation")
    mask2, prediction2 = grad_cam2(input, target_index)
    show_cam_on_image(img, mask2, path, "rotation")

    return JSONResponse({'result1': str(prediction1),'result2': str(prediction2)})

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5042)
