#!/bin/bash

output=`$1`

dmass=`echo "$output" | grep d_mass | awk '{print $2}'`
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

dte=`echo "$output" | grep d_te | awk '{print $2}'`
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

