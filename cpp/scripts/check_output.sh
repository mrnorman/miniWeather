#!/bin/bash

showusage() {
  printf "Usage:\n"
  printf "./check_output.sh  executable  mass_relative_tolerance  energy_relative_tolerance\n\n"
}
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  showusage
  exit 0
fi
if [[ "$1" == "" ]]; then
  printf "ERROR: Did not specify an executable\n\n"
  showusage
  exit -1
fi
if [[ "$2" == "" ]]; then
  printf "ERROR: Did not specify a mass tolerance\n\n"
  showusage
  exit -1
fi
if [[ "$3" == "" ]]; then
  printf "ERROR: Did not specify a total energy tolerance\n\n"
  showusage
  exit -1
fi

if [[ ! -f $1 ]]; then
  exit -1
fi
${TEST_MPI_COMMAND} $1 > ${1}.output || exit -1

dmass=`grep d_mass ${1}.output | awk '{print $2}'`
if [[ "$dmass" == "NaN" ]]; then
  printf "Mass change is NaN\n\n"
  exit -1
fi
if (( `awk "BEGIN {val=$dmass; abs=val>0?val:-val; ret=abs<$2?0:-1; print ret}"` )); then
  printf "ERROR: Mass change magnitude is too large\n\n"
  printf "Change in mass: $dmass"
  printf "Mass tolerance: $2"
  exit -1
fi

dte=`grep d_te ${1}.output | awk '{print $2}'`
if [[ "$dte" == "NaN" ]]; then
  printf "Total energy change is NaN\n\n"
  exit -1
fi
if (( `awk "BEGIN {val=$dte; ret=val<0?0:-1; print ret}"` )); then
  printf "ERROR: Total energy change must be negative\n\n"
  exit -1
fi
if (( `awk "BEGIN {val=$dte; abs=val>0?val:-val; ret=abs<$3?0:-1; print ret}"` )); then
  printf "ERROR: Total energy change magnitude is too large\n\n"
  printf "Change in total energy: $dte"
  printf "Total energy tolerance: $3"
  exit -1
fi

rm -f ${1}.output

