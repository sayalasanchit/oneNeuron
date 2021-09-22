from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging
import os

logging_str="[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format=logging_str, filename=os.path.join(log_dir, "running_logs.log"), filemode='a')


def main(AND, eta, epochs, filename, plotname): 
    df=pd.DataFrame(AND)
    logging.info(f"Dataframe:\n{df}")
    X, y=prepare_data(df)
    model=Perceptron(eta, epochs)
    model.fit(X, y)
    _=model.totalLoss()
    save_model(model, filename)
    save_plot(df, plotname, model)

if __name__=="__main__":
    AND={
        'x1':[0, 0, 1, 1],
        'x2':[0, 1, 0, 1],
        'y':[0, 0, 0, 1],
    }
    ETA=0.3
    EPOCHS=10
    try:
        logging.info(">>>>>>  STARTING TRAINING  <<<<<<")
        main(AND, ETA, EPOCHS, "and.model", "and.png")
        logging.info(">>>>>>  TRAINING DONE SUCCESSFULLY  <<<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e