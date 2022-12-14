# CLI for prediction
- `predict` function  
  - input format: dataframe  
  - `curl`  
    - `bentoml serve IrisClassifier:latest`  
    - `curl -i --header "Content-Type: application/json" --request POST --data '[[5.1, 3.5, 1.4, 0.2]]' http://localhost:5000/predict`  
  - `bentoml` cli  
    - `bentoml run IrisClassifier:latest predict --input '[[5.1, 3.5, 1.4, 0.2]]'`  


- `predict_json` function
  - input format: json  
  - `curl`  
    - `bentoml serve IrisClassifier:latest`  
    - `curl -i --header "Content-Type: application/json" --request POST --data '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}' http://localhost:5000/predict_json`  
  - `bentoml` cli  
    - `bentoml run IrisClassifier:latest predict_json --input '{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2}'`  


- **workflow**  
  - `main.py`, `iris_classifier.py`(파일명은 모델에 따라 가변적) 코드 작성  
  - `main.py` 실행 → `~/bentoml/repository/{service_name}/{service_version}` 경로에 docker image build를 위한 파일들 생성  
  - 테스트 1: `bentoml serve` 명령어로 모델을 local 5000 port에 올린 후 `curl` 명령어로 request   
  - docker build  
    - working directory를 `~/bentoml/repository/{service_name}/{service_version}` 경로로 맞춘다  
    - working directory에서 터미널을 켜고 `docker build -t {image_name}:{image_tag} .` 명령어 입력  
    - 위 명령어가 실행되는 중에 에러가 발생하면 python libraries 간 dependency 문제일 수 있음  
    - bentoml이 생성한 `requirements.txt`를 확인해보면 실마리를 얻을 수 있음   
  - 테스트 2: `docker run` 명령어로 모델을 local 5000 port에 올린 후 `curl` 명령어로 request  
  - 테스트 결과 이상 없으면 아래로 넘어가고 이상 있으면 코드 수정 후 다시 docker build  
  - aws login  
    - `aws configure` → Access key ID, Secret access key 등 입력  
    - AWS ECR login  
  - `docker tag`  
  - `docker push`  
