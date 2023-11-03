from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import models.networks as networks
from torchcam.methods import CAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BreaKHis(Dataset):

    def __init__(self, img_list, transform = None):
        self.transform = transform
        self.img_list = img_list

    def __getitem__(self, index):
        path = self.img_list[index]

        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return path, img

    def __len__(self):
        return len(self.img_list)


class BaseBackendModel():

    def __init__(self, reject_threshold=0.7):
        self.reject_threshold = reject_threshold

    def inference(self, img_path):
        raise NotImplementedError


    @staticmethod
    def get_label(task, id, abbrev=False):
        assert task in ['binary', 'subtype'], 'task should be either binary or subtype'
        labels = BaseBackendModel.get_all_labels(task, abbrev)
        if id is None:
            return 'reject'
        if task == 'binary':
            return labels[id] if id < 2 else 'reject'
        else:
            return labels[id] if id < 8 else 'reject'
        
    
    @staticmethod
    def get_all_labels(task, abbrev=False):
        assert task in ['binary', 'subtype'], 'task should be either binary or subtype'
        if abbrev:
            bin_list = ['B', 'M']
            subtype_list = ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']
        else:
            bin_list = ['Benign', 'Malignant']
            subtype_list = ['Adenosis', 'Fibroadenoma', 'Phyllodes Tumor', 'Tubular Adenoma', 'Ductal Carcinoma', 'Lobular Carcinoma', 'Mucinous Carcinoma', 'Papillary Carcinoma']

        return bin_list if task == 'binary' else subtype_list
    

    @staticmethod
    def generate_empty_result():
        return {'pred':{'binary':None, 'subtype':None}, 'prob':{'binary':[0,0], 'subtype':[0,0,0,0,0,0,0,0]}, 'cam':{'binary':None, 'subtype':None}}
    

class BackendModel(BaseBackendModel):

    data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.7862, 0.6261, 0.7654), (0.1065, 0.1396, 0.0910)), # BreakHis normalization
                transforms.Resize((460, 700), antialias=True)
            ]
        )

    def __init__(self, reject_threshold=0.7):
        super().__init__(reject_threshold)

        self._models = {
            'binary': networks.ResNet50(num_classes=2),
            'subtype': networks.DenseNet201(num_classes=8),
        }
        self._ckpts = {
            'binary': './models/ckpt/resnet50-bin.pth',
            'subtype': './models/ckpt/densenet201-sub.pth',
        }
        self.loaded = False

    
    def _load(self):
        if self.loaded:
            return
        self.loaded = True
        for task_type in self._models.keys():
            self._models[task_type].load_state_dict(torch.load(self._ckpts[task_type])['model_state_dict'])
            self._models[task_type] = torch.nn.Sequential(self._models[task_type], torch.nn.Softmax(dim=1))
            self._models[task_type].to(device)
            self._models[task_type].eval()


    def inference(self, img_path):
        self._load()
        dataset = BreaKHis(img_path, transform=self.data_transform)
        iterator = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        cam_extractors = {
            'binary': CAM(self._models['binary']),
            'subtype': CAM(self._models['subtype'])
        }
        img_path = []
        binary_outputs = torch.tensor([]).to(device)
        subtype_outputs = torch.tensor([]).to(device)
        binary_cams = torch.tensor([]).to(device)
        subtype_cams = torch.tensor([]).to(device)
        with torch.no_grad():
            for path, img in iterator:
                img_tensor = img.to(device)
                binary_output = self._models['binary'](img_tensor)
                subtype_output = self._models['subtype'](img_tensor)

                img_path += path
                binary_outputs = torch.cat((binary_outputs, binary_output), dim=0)
                subtype_outputs = torch.cat((subtype_outputs, subtype_output), dim=0)
                
                binary_cam = cam_extractors['binary'](torch.argmax(binary_output, dim=1).tolist(), binary_output)[0]
                subtype_cam = cam_extractors['subtype'](torch.argmax(subtype_output, dim=1).tolist(), subtype_output)[0]
                binary_cams = torch.cat((binary_cams, binary_cam), dim=0)
                subtype_cams = torch.cat((subtype_cams, subtype_cam), dim=0)
        
        binary_outputs = binary_outputs.cpu()
        subtype_outputs = subtype_outputs.cpu()
        binary_maxes, binary_argmaxes = torch.max(binary_outputs, dim=1)
        subtype_maxes, subtype_argmaxes = torch.max(subtype_outputs, dim=1)

        results = {}
        for path, binary_output, subtype_output, binary_max, binary_argmax, subtype_max, subtype_argmax, binary_cam, subtype_cam\
            in zip(img_path, binary_outputs, subtype_outputs, binary_maxes, binary_argmaxes, subtype_maxes, subtype_argmaxes, binary_cams, subtype_cams):
            results[path] = {'pred':{}, 'prob':{}, 'cam':{}}
            if binary_max < self.reject_threshold:
                results[path]['pred']['binary'] = None
            else:
                results[path]['pred']['binary'] = binary_argmax.item()
            if subtype_max < self.reject_threshold:
                results[path]['pred']['subtype'] = None
            else:
                results[path]['pred']['subtype'] = subtype_argmax.item()
            results[path]['cam']['binary'] = binary_cam.cpu().numpy()
            results[path]['prob']['binary'] = binary_output.tolist()
            results[path]['cam']['subtype'] = subtype_cam.cpu().numpy()
            results[path]['prob']['subtype'] = subtype_output.tolist()
        return results


# Used for testing
class RandomBackendModel(BaseBackendModel):

    def __init__(self, reject_threshold=0.7):
        super().__init__(reject_threshold)

    def inference(self, img_path):
        import numpy as np
        import time
        time.sleep(10)
        result = {}
        for path in img_path:
            result[path] = {'pred':{}, 'prob':{}, 'cam':{}}
            result[path]['prob']['binary'] = np.random.dirichlet(np.ones(2), size=1)[0]
            result[path]['prob']['subtype'] = np.random.dirichlet(np.ones(8), size=1)[0]
            if np.max(result[path]['prob']['binary']) < self.reject_threshold:
                result[path]['pred']['binary'] = None
            else:
                result[path]['pred']['binary'] = np.argmax(result[path]['prob']['binary'])
            if np.max(result[path]['prob']['subtype']) < self.reject_threshold:
                result[path]['pred']['subtype'] = None
            else:
                result[path]['pred']['subtype'] = np.argmax(result[path]['prob']['subtype'])
            result[path]['prob']['binary'] = result[path]['prob']['binary'].tolist()
            result[path]['prob']['subtype'] = result[path]['prob']['subtype'].tolist()
            result[path]['cam']['binary'] = np.random.rand(460, 700)
            result[path]['cam']['subtype'] = np.random.rand(460, 700)

        return result
    