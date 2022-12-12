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