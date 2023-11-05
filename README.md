# Hairthon
# 프로젝트 발표 영상(.txt, 문자열 블러 처리)

[![Video Label](/src/imgs/logo.png)]([https://youtu.be/9pCCKXYSrt8?si=SUoRJZ16m9R-fcXL](https://www.youtube.com/watch?v=g-hA8vYcROk#t=31m54s))

##### 이미지를 누르면 시연영상을 확인할 수 있습니다.

# 프로젝트 시연 영상(.wav, 음성![image](https://github.com/Dandyoung/Hairthon/assets/109204274/7a5b7886-dbda-43dc-886f-01bb96579fa3)
 블러 처리)

[결과 영상](https://youtu.be/VRiUKeyz3M4)

<br><br>

## ✔목차
* [프로젝트 정보](#🔎프로젝트-정보)
* [프로젝트 소개](#🖐프로젝트-소개)
* [팀원 소개](#🙋‍♀️팀원-소개)
* [모델](#모델)
* [데이터](#데이터)
* [학습 과정](#학습-과정)
* [Ref](#📝ref)

<br><br>

## 🔎프로젝트 정보
> 원티드X유데미X조코딩 AI 해커톤
> 
> 개발 기간: 2023.7.22 ~ 2022.8.5 (2주) 

<br><br>

## 🖐프로젝트 소개
> 본 프로젝트는 2023 공개SW 원티드X유데미X조코딩 AI 해커톤 출품작으로, semyir은 인공지능 모델로 개인의 얼굴형을 분석하거나, 본인이 원하는 머리스타일과 합성해볼 수 있는 웹 서비스입니다.
> 다양한 머리를 시도하기 좋아하는 분들, 평소에 자신의 얼굴에 어울리는 머리스타일을 찾던 사람들을 위한 실용적인 솔루션입니다.

<br><br>
> 참여 인원 : 5인(백앤드 2명, 프론트앤드 2명, AI 앤지니어 2명)
> 기술 스택 : Python(Flask), React(Next.js), Tensorflow, Pytorch, OpenCV, Figma
> 
## 🙋‍♀️팀원 소개

<br><br><br>
# 주요 기능
1. **얼굴형 분석**: 사용자의 얼굴 사진을 업로드하면, 사용자의 윤곽, 이마 크기, 턱선 형태 등을 고려해 얼굴형을 분석합니다. 이를 통해 사용자의 얼굴형과 어올리는 헤어스타일에 대한 정보를 제공합니다.
<br>
2. **원하는 머리스타일 합성:** 사용자가 원하는 다양한 헤어스타일을 자연스럽게 가상으로 시착할 수 있는 시뮬레이션을 제공합니다. 이를 통해 사용자는 헤어스타일을 실제로 시도해보기 전에 직접 자신의 얼굴에 어울리는 여러 헤어스타일을 비교 분석할 수 있습니다.
<br>
3. **이미지 해상도 업스케일링** : 사용자가 저화질의 이미지를 업로드하여도, 만족도 높은 결과를 위해 x4배 해상도를 업스케일링하여 합성모델에 넣어줍니다.

# 사용 모델
- 얼굴형 분석 모델: 얼굴형 분석은 빠르면서 정량적으로 좋은 [ShuffleNetV2](https://github.com/Randl/ShuffleNetV2-pytorch) 에서 참고하여 학습했습니다.
<br>
- 얼굴 헤어스타일 합성 모델: 원하는 머리스타일과 내 얼굴을 합성할 땐, 얼굴의 이미지에서 얼굴의 형태와 머리스타일의 데이터를 stylegan2로 재생성하는 ****[Hairstyle Transfer between Face Images](https://cmp.felk.cvut.cz/hairstyles/)****의 논문을 사용하였습니다.
<br>
- 이미지 업스케일링 모델: 얼굴 헤어스타일 합성 모델에서의 실험결과, 좋은 해상도의 이미지 일수록 좋은 합성결과를 가져오기에 [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN)을 통해, 합성모델에서의 좋은 결과를 가져오고자 했습니다.


더 자세한 내용은 [코드](https://github.com/DAUOpenSW/PVMM/blob/main/src/models.py)를 참고해 주세요.

# 데이터

기존의 데이터는 Face Shape Dataset 데이터를 사용하였습니다. 이후, 한국인 얼굴에 대한 일반화시키기위해 Kaggle의 데이터 세트로 학습 데이터를 변경하여 모델 학습을 진행했습니다


## 📝Ref
프론트앤드 배포 : 
