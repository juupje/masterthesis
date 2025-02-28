#for N in 1 2 10 20 40 128; do
#    echo "N = $N";
#    python3 preprocessing.py -n 1 --chunksize 20000 -N $N --signal 0 "$DATA_DIR/transformer/raw/qcd_test_nsamples200000_trunc_5000_nonan.h5" "$DATA_DIR/transformer/N${N}_qcd_test_nonan.h5";
#    python3 preprocessing.py -n 1 --chunksize 20000 -N $N --signal 1 "$DATA_DIR/transformer/raw/top_test_nsamples200000_trunc_5000_nonan.h5" "$DATA_DIR/transformer/N${N}_top_test_nonan.h5";
#    python3 preprocessing.py -n 1 --chunksize 20000 -N $N --signal 0 "$DATA_DIR/transformer/raw/qcd_train_nsamples1000000_trunc_5000_nonan.h5" "$DATA_DIR/transformer/N${N}_qcd_train_nonan.h5";
#    python3 preprocessing.py -n 1 --chunksize 20000 -N $N --signal 1 "$DATA_DIR/transformer/raw/top_train_nsamples1000000_trunc_5000_nonan.h5" "$DATA_DIR/transformer/N${N}_top_train_nonan.h5";
#done

#DATA_DIR=$DATA_DIR/transformer/original_binned
#python3 restore_bins.py --chunksize 16384 "$DATA_DIR/ZJetsToNuNu_test___1Mfromeach_403030.h5" "$DATA_DIR/qcd_test_unbinned.h5" --bin-dir $DATA_DIR
#python3 restore_bins.py --chunksize 16384 "$DATA_DIR/ZJetsToNuNu_train___1Mfromeach_403030.h5" "$DATA_DIR/qcd_train_unbinned.h5" --bin-dir $DATA_DIR
#python3 restore_bins.py --chunksize 16384 "$DATA_DIR/TTBar_test___1Mfromeach_403030.h5" "$DATA_DIR/top_test_unbinned.h5" --bin-dir $DATA_DIR
#python3 restore_bins.py --chunksize 16384 "$DATA_DIR/TTBar_train___1Mfromeach_403030.h5" "$DATA_DIR/top_train_unbinned.h5" --bin-dir $DATA_DIR

#python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 0 "$DATA_DIR/qcd_test_unbinned.h5" "$DATA_DIR/qcd_test_nonan.h5"
#python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 0 "$DATA_DIR/qcd_train_unbinned.h5" "$DATA_DIR/qcd_train_nonan.h5"
#python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 1 "$DATA_DIR/top_test_unbinned.h5" "$DATA_DIR/top_test_nonan.h5"
#python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 1 "$DATA_DIR/top_train_unbinned.h5" "$DATA_DIR/top_train_nonan.h5"

DATA_DIR=$DATA_DIR/transformer/10M
#for i in {0..4}; do
#    python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 1 "$DATA_DIR/top_$i.h5" "$DATA_DIR/top_${i}_pre.h5";
#    python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 0 "$DATA_DIR/qcd_$i.h5" "$DATA_DIR/qcd_${i}_pre.h5";
#done

python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 1 "$DATA_DIR/top_test_nonan.h5" "$DATA_DIR/top_test_pre.h5";
python3 preprocessing.py -n 1 -N 128 --chunksize 20000 --signal 0 "$DATA_DIR/qcd_test_nonan.h5" "$DATA_DIR/qcd_test_pre.h5";
