echo
echo python train_model.py $@ --both
echo
python train_model.py $@ --both

echo
echo python test_model.py $@ --target data/eTest.csv
echo
python test_model.py $@ --target data/eTest.csv
