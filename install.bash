#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Add required paths to users .bashrc file
echo "Updating your .bashrc file to include the paths required by the TFMin library"

echo "" >> ${HOME}/.bashrc
echo "# setup paths for TFMin, the minimal Tensorflow library" >> ${HOME}/.bashrc

echo "export PYTHONPATH=${SCRIPT_PATH}/python:\$PYTHONPATH" >> ${HOME}/.bashrc
echo "export CPLUS_INCLUDE_PATH=${SCRIPT_PATH}/src/include:\$CPLUS_INCLUDE_PATH" >> ${HOME}/.bashrc

# update current environment variables
echo "Updating current environment variables"
export PYTHONPATH=${SCRIPT_PATH}/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=${SCRIPT_PATH}/src/include:$CPLUS_INCLUDE_PATH

echo "TFMin installation complete."
