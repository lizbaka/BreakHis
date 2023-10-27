import json
from models.inference import BackendModel, RandomBackendModel


def main():
    # ./imgs/img (1).png, ./imgs/img (2).png, ..., ./imgs/img (29).png
    imglist = [f'./imgs/img ({i}).png' for i in range(1, 30)]
    # random generate for testing, with 10s delay
    backend = RandomBackendModel(reject_threshold=0.7)
    # neural network model
    # backend = BackendModel(reject_threshold=0.7)
    result = backend.inference(imglist)

    print('binary classification for ./imgs/img (24).png')
    if result['./imgs/img (24).png']['pred']['binary'] is None:
        print('reject')
    else:
        print(result['./imgs/img (24).png']['pred']['binary'])
        print(BackendModel.get_label('subtype', result['./imgs/img (24).png']['pred']['binary']))
    print(result['./imgs/img (24).png']['prob']['binary'])

    print('subtype classification for ./imgs/img (24).png')
    if result['./imgs/img (24).png']['pred']['subtype'] is None:
        print('reject')
    else:
        print(result['./imgs/img (24).png']['pred']['subtype'])
        print(BackendModel.get_label('subtype', result['./imgs/img (24).png']['pred']['subtype']))
    print(result['./imgs/img (24).png']['prob']['subtype'])


if __name__ == '__main__':
    main()