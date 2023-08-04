import cv2
from facenet_pytorch import MTCNN
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device,select_largest=False)

img=cv2.imread("1.jpg")

boxes,prob=mtcnn.detect(img,landmarks=False)

boxes=boxes.astype(int)
print(boxes)
max_box=prob.argmax()
quarter_high=(boxes[max_box][3]-boxes[max_box][1])//4
quarter_side=(boxes[max_box][2]-boxes[max_box][0])//3

crop_img=img[boxes[max_box][1]-quarter_high:boxes[max_box][3]+quarter_high//2,boxes[max_box][0]-quarter_side:boxes[max_box][2]+quarter_side]

resize_img=cv2.resize(crop_img,dsize=(1024,1024))
cv2.imwrite("1.png",resize_img)


cv2.imshow("resize_img",crop_img)
cv2.waitKey()
cv2.destroyAllWindows()

