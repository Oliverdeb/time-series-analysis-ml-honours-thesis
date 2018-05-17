./clean.sh
python3 create_sets.py
python3 model.py
python3 forecast.py
gsutil cp out.png gs://eggsy
