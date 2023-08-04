import torch
import h5py

def convert_pth_to_h5(pth_file, h5_file):
    # PyTorch 모델 로드
    model = torch.load(pth_file,map_location=torch.device('cpu'))
    
    # h5 파일 생성 및 모델 파라미터 저장
    with h5py.File(h5_file, 'w') as hf:
        for name, param in model.items():
            hf.create_dataset(name, data=param.numpy())

if __name__ == "__main__":
    pth_file = "CustomArc_efficientnet.pth"  # 변환할 pth 파일 경로 입력
    h5_file = "model.h5"  # 변환된 h5 파일 저장 경로 입력

    convert_pth_to_h5(pth_file, h5_file)