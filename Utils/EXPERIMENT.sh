source activate resproj

pwd=$PWD
replace="Programs"
pwd=${pwd/Utils/$replace}

printf "Running TFIDF_model.py\n"
python $pwd/TFIDF_model.py

printf "Running GloVe_model.py\n"
python $pwd/GloVe_model.py

conda deactivate