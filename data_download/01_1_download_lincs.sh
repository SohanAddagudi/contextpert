echo "Downloading LINCS data to $CONTEXTPERT_RAW_DATA_DIR/lincs/"
# Level 3 expression data
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel3%5FINF%5Fmlr12k%5Fn1319138x12328.gctx.gz
# Landmark gene info
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo%5Fdelta%5Flandmark.txt.gz
# Perturbation info (contains smiles)
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo.txt.gz
# Inst info
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Finst%5Finfo.txt.gz
# Sig metrics
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Fmetrics.txt.gza
# Sig info
wget -P $CONTEXTPERT_RAW_DATA_DIR/lincs/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo.txt.gz
# Unzip all
echo "Unzipping files, this may take a moment..."
gunzip -v $CONTEXTPERT_RAW_DATA_DIR/lincs/*.gz
# Copy pert info into working data dir
cp $CONTEXTPERT_RAW_DATA_DIR/lincs/GSE92742_Broad_LINCS_pert_info.txt $CONTEXTPERT_DATA_DIR/GSE92742_Broad_LINCS_pert_info.txt