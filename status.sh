#! /bin/bash
name="$1"
echo Searching with $name

find . -name "*${name}_scheduled" -or -name "*${name}_started" -or -name "*${name}_complete" | grep -v download | while read line; do
  jobid=$(cat $line);
  sacct --noheader -j ${jobid} --format=JobID,Jobname%30,State,Start,End,Elapsed,Time | grep -v '\.bat+' | grep -v '\.ext' | tail -n 1;
done
