DATA_DIR="/home/joep/Documents/lhco_data"
python3 preprocessing_lhco.py -n 6 --chunksize 8192 --events  20000 --offset 1000000 -N 100 --mjj 3300:3700 --no-rotate "$DATA_DIR/events_anomalydetection_v2.h5" "$DATA_DIR/mono_sn_test_SR.h5" --sequential
python3 preprocessing_lhco.py -n 6 --chunksize 8192 --events  80000 --offset 1020000 -N 100 --mjj 3300:3700 --no-rotate "$DATA_DIR/events_anomalydetection_v2.h5" "$DATA_DIR/mono_sn_train_SR.h5" --sequential
python3 preprocessing_lhco.py -n 8 --chunksize 8192 --events 100000                  -N 100 --mjj 3300:3700 --no-rotate "$DATA_DIR/events_anomalydetection_v2.h5" "$DATA_DIR/mono_bg_test_SR.h5" --sequential
python3 preprocessing_lhco.py -n 8 --chunksize 8192 --events 900000 --offset  100000 -N 100 --mjj 3300:3700 --no-rotate "$DATA_DIR/events_anomalydetection_v2.h5" "$DATA_DIR/mono_bg_train_SR.h5" --sequential