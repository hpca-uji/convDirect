#!/bin/bash

# RUN_PATH="$(pwd -P)"

SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit
  pwd -P
)"

# The next path should be the same as the output pat of convDirect_eval.sh
OUTPUT_PATH=./convdirect_output_$(hostname)

ALGORITHMS=(
  "convdirect_im2row_nhwc_default"
  "convdirect_conv_gemm_nhwc_default"
  "convdirect_block_blis_nhwc_blis"
  "convdirect_block_blis_nhwc_8x12"
  "convdirect_block_blis_nhwc_4x20"
  "convdirect_block_shalom_nhwc_7x12_npa_u4"
  "convdirect_tzemeng_nhwc_7x12_u4"
)

for NET in "${SCRIPT_PATH}"/cnn/*; do
  for ALGORITHM in "${ALGORITHMS[@]}"; do
      # convdirect_eval.sh writes its outputs to ${RUN_DIR}/runs/CNN_-_algorithm.csv
      # TMIN=0 TEST=T \
      "${SCRIPT_PATH}"/convDirect_eval.sh "${ALGORITHM}" "${NET}"
  done
done

cat <<EOF


EVALUATION COMPLETED!

If you want to process the results locally, please execute the following command:
  ${SCRIPT_PATH}/convDirect_eval_all_process.py ${OUTPUT_PATH}

Please note that the evaluation results can be processed on another machine.
Just copy to that machine:
 - the 'convDirect_eval_all_process.py' script  and
 - the '${OUTPUT_PATH}' directory,
then, execute there:
  ./convDirect_eval_all_process.py "${OUTPUT_PATH}"

EOF
