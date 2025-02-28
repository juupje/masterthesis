#python3 preprocessing.py -N 30 -n 8 --events 1000000 $DATA_DIR/lhco/raw/events_anomalydetection_v2.h5 $DATA_DIR/lhco/original/N30-bg.h5 --split 0.5:train 0.5:val
#python3 preprocessing.py -N 30 -n 8 --events 100000 --offset 1000000 $DATA_DIR/lhco/raw/events_anomalydetection_v2.h5 $DATA_DIR/lhco/original/N30-sn.h5 --split 0.5:train 0.5:val

#python3 preprocessing.py -N 30 -n 8 --events 100000 $DATA_DIR/lhco/raw/DP_events_QCD2-8.h5 $DATA_DIR/lhco/new/N30-bg-test.h5
#python3 preprocessing.py -N 30 -n 8 --events 100000 --key Particles $DATA_DIR/lhco/raw/DP_events_signal.h5 $DATA_DIR/lhco/new/N30-sn-test.h5

for i in {1..5}; do
python3 preprocessing.py -N 100 -n 8 --key Particles $DATA_DIR/DelphesPythia/HDF/pt500/dp_events_qcd_pt500-$i.h5 $DATA_DIR/lhco/new/pt500/N100-bg-$i.h5
done
