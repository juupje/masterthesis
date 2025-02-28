#DATA_DIR="/media/joep/Hard Drive/thesis_data"
python3 preprocessing_toptagging.py -n 4 --chunksize 20000 --events 220000 -N 100 "$DATA_DIR/toptagging/raw/train.h5" "$DATA_DIR/toptagging/train.h5"
#python3 preprocessing_toptagging.py -n 4 --chunksize 20000 --events 200000 -N 100 "$DATA_DIR/toptagging/raw/test.h5" "$DATA_DIR/toptagging/test.h5"
python3 preprocessing_toptagging.py -n 4 --chunksize 20000 --events 200000 -N 100 "$DATA_DIR/toptagging/raw/val.h5" "$DATA_DIR/toptagging/val.h5"