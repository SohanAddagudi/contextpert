python 04_2_transcriptome_embeddings.py --backbone "aido_cell_3m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_cp_smiles_qc.csv
python 04_2_transcriptome_embeddings.py --backbone "aido_cell_3m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_sh_qc.csv

python 04_2_transcriptome_embeddings.py --backbone "aido_cell_10m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_cp_smiles_qc.csv
python 04_2_transcriptome_embeddings.py --backbone "aido_cell_10m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_sh_qc.csv

python 04_2_transcriptome_embeddings.py --backbone "aido_cell_100m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_cp_smiles_qc.csv
python 04_2_transcriptome_embeddings.py --backbone "aido_cell_100m" --chunk-size 50000 --input $CONTEXTPERT_DATA_DIR/trt_sh_qc.csv