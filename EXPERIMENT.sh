source activate resproj

printf "Running TFIDF_model.py\n"
python TFIDF_model.py

printf "Running GloVe_model.py\n"
python GloVe_model.py

conda deactivate