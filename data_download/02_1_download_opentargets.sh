echo "Downloading OpenTargets data to $CONTEXTPERT_RAW_DATA_DIR/opentargets/"
export OT_REL=25.06
export OT_WORK=$CONTEXTPERT_RAW_DATA_DIR/opentargets/
mkdir -p "$OT_WORK/$OT_REL"

for ds in disease association_by_datasource_direct known_drug; do
  rsync -rpltvz --delete                     \
        rsync.ebi.ac.uk::pub/databases/opentargets/platform/$OT_REL/output/$ds \
        "$OT_WORK/$OT_REL/"
done