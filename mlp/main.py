
import numpy as np
import visualisation
import model_functions as modf
import preprocessing
import config
import build
import pickle

np.random.seed(8888)

model_type = config.model

if __name__ == '__main__':
    df = preprocessing.download_data()
    df = preprocessing.process_data(df)

    model = build.create_model()

    build.build_model(model, df)
