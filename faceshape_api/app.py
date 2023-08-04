from module import *
class_labels = ['하트형/역삼각형 얼굴', '긴 얼굴', '계란형 얼굴', '둥근형 얼굴', '네모형/각진얼굴']


###여기부터 수정
## 1) cors 처리 해야함 https://hairthon.vercel.app/
application = Flask(__name__)
application.config['DEBUG'] = True
CORS(application)

@application.route("/dl_Img",methods=['POST'])
def dl_Img():
    try:
        parsed_request = request.files.get('file')
        fileName = request.form.get('fileName')
        image_data = parsed_request.read()
        # 2) react에서 이미지를 받아올 수 있게.
        model_checkpoint_path = "mobilenet74.pt"  # 모델 체크포인트 경로 설정
        # 모델 적용
        model = load_model(model_checkpoint_path)
        input_tensor = preprocess_image(image_data)
        prediction = predict_image(model, input_tensor)
        pred_list = tensor_to_list(prediction)
        
        real_list = [round(element,3) for row in pred_list for element in row]
        # 3) return 해줄떄 key, value로바꿔서
        result_dic = dict(zip(class_labels, real_list))
        result = json.dumps(result_dic, ensure_ascii=False)
        return result
    except Exception as e:
        print(str(e))
        return {"error": str(e)}

    
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80)