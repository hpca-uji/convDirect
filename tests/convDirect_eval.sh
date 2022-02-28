#!/bin/bash

#********************************************************#
#   ********** EVAL CONFIGURATION VARIABLES **********   #
#********************************************************#
# [*] TMIN: Minimum Execution Time for each convolution.
#           If TEST activated, TMIN value must be 0.0
TMIN=${TMIN:-2.0}
#--------------------------------------------------------#
# [*] TEST: Activate for convolution results evaluation.
#           [ T: Enable ], [ F: Disable ]
TEST=${TEST:-F}
#--------------------------------------------------------#
# [*] DEBUG: Activate for debug mode. Prints matrix values.
#           [ T: Enable ], [ F: Disable ]
DEBUG=${DEBUG:-F}
#********************************************************#
#********************************************************#
#********************************************************#

RUN_PATH="$(pwd -P)"

SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

my_help() {
  cat <<EOF
Usage: convDirect_eval.sh ALGORITHM_NAME CONFIG_FILENAME

Calls convDirect_eval and provides it with default values
for the rest of the arguments it requires.

The required parameters for this script are:
 + ALGORITHM_NAME: The algorithm to be evaluated.
 + CONFIG_FILENAME: The configuration filename that
                    describes the evaluations to be
                    performed.

You can execute convDirect_info to obtain a list of supported
algorithms.

Examples of configuration evaluations can be found on the
directories tests/cnn/ and tests/batch/.
EOF
}

if [ "$#" -ne 2 ]; then
  my_help
  echo
  echo "Error: Wrong number of parameters"
  exit 1
fi

ALGORITHM_NAME=${1}
CONFIG_PATH=${2}
CONFIG_FILENAME=$(basename "${2}")
OUTPUT_PATH=./convdirect_output_$(hostname)

if [ ! -f "${CONFIG_PATH}" ]; then
  my_help
  echo
  echo "ERROR: The configuration file '${CONFIG_PATH}' does not exist. Please, enter a valid filename."
  exit 1
fi

echo
echo "Starting evaluation script for direct convolution..."

[ -d "${OUTPUT_PATH}" ] || mkdir "${OUTPUT_PATH}"

OUTPUT_FILENAME="${OUTPUT_PATH}/${CONFIG_FILENAME}_-_${ALGORITHM_NAME}.csv"

export OMP_NUM_THREADS=1
export OMP_BIND=true

if [[ "${CONFIG_PATH}" == *"cnn"* ]]; then
  CNN_OR_BATCH="cnn"
else
  CNN_OR_BATCH="batch"
fi

export PATH=${PATH}:./tests:../build/tests

convDirect_eval "${ALGORITHM_NAME}" "${CONFIG_PATH}" \
  "${CNN_OR_BATCH}" \
  ${TMIN} \
  ${TEST} \
  ${DEBUG} \
  "${OUTPUT_FILENAME}"

# Write host information
python > "${OUTPUT_PATH}/host.nfo" <<EOF
import platform
h = platform.node().capitalize()
p = platform.processor().replace(' Core Processor', '').replace('(', '').replace(')', '')
print('{} ({})'.format(h, p))
EOF
