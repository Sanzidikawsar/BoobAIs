#TASS Movidius Trainer
pip3 install -r requirements.txt
python3 TassMovidiusData.py sort
python3 TassMovidiusTrainer.py
mvNCCompile model/MovidiusInception.pb -in=input -on=InceptionV3/Predictions/Softmax
mv graph igraph
python3 TassMovidiusEval.py 
python3 TassMovidiusClassifier.py InceptionTest
