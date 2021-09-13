from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd

def main(AND, eta, epochs, filename, plotname):
    df=pd.DataFrame(AND)
    print(f"Dataframe:\n{df}")
    X, y=prepare_data(df)
    model=Perceptron(eta, epochs)
    model.fit(X, y)
    _=model.totalLoss()
    save_model(model, filename)
    save_plot(df, plotname, model)

if __name__=="__main__":
    OR={
        'x1':[0, 0, 1, 1],
        'x2':[0, 1, 0, 1],
        'y':[0, 1, 1, 1],
    }
    ETA=0.3
    EPOCHS=10
    main(OR, ETA, EPOCHS, "or.model", "or.png")