import scripts.normalization.earlier.functions as func
import pandas as pd

pd.set_option('display.max_row', None)

file_path = "../../data/morris/Muticlass_csv_1.csv"

data = pd.read_csv(file_path)

col=data.columns
func.op_csvs(col)



