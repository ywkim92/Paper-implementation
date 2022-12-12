import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

answer_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        ans = self.artifacts.model.predict(df)
        return ans

    @api(input=JsonInput(), batch=False)
    def predict_json(self, js):
        print(js)
        print()
        df = list(map(float, js.values()))
        print(df)
        ans = self.artifacts.model.predict([df])
        ans = answer_dict[ans[0]]
        return ans