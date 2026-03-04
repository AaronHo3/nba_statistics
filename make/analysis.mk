.PHONY: notebook eda

notebook: setup data-dir ## Launch Jupyter Notebook
	$(JUPYTER) notebook

eda: setup validate-data ## Quick dataset profile in terminal
	$(PY) -c "from pathlib import Path; import pandas as pd; p=sorted(Path('$(DATA_DIR)').glob('*.csv')); print(f'Files: {len(p)}'); [print('-',x.name) for x in p]; df=pd.read_csv(p[0]); print('\nUsing:',p[0].name); print('Rows:',len(df)); print('Cols:',len(df.columns)); print('\nColumns:'); [print('-',c) for c in df.columns]; print('\nTop missing values:'); print(df.isna().sum().sort_values(ascending=False).head(15).to_string())"

