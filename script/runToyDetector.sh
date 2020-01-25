# ./main [fileName] [dc] [deltao] [deltac] [rhoc] [useGPU] [totalNumberOfEvent] [verbose]
# ./mainCuplaCPUTBB toyDetector_10000 3 5 5 8 1 1 0
# ./main toyDetector_10000 3 5 5 8 0 1 0
# sh script/runToyDetector.sh &> log/ryzen_toyDetector.log
# sh script/runToyDetector.sh &> log/patatrack02_toyDetector.log

export DC=3
export DELTAO=5
export DELTAC=5
export RHOC=8
export NEVENT=200


echo "----------------"
echo "running with CPU"
echo "----------------"
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  ./main toyDetector_$i $DC $DELTAO $DELTAC $RHOC 0 $NEVENT 0
done

echo "----------------"
echo "running with GPU"
echo "----------------"
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  ./main toyDetector_$i $DC $DELTAO $DELTAC $RHOC 1 $NEVENT 0
done


echo "----------------"
echo "running with GPU with nvprof"
echo "----------------"
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  nvprof ./main toyDetector_$i $DC $DELTAO $DELTAC $RHOC 1 $NEVENT 0
done


for tbbnthreads in 1 4 8 16
# for tbbnthreads in 1 10 20 40
do 

  echo "----------------"
  echo "running with CUPLA CPU TBB nthreads=" $tbbnthreads
  echo "----------------"
  for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
  do
    ./mainCuplaCPUTBB toyDetector_$i $DC $DELTAO $DELTAC $RHOC 1 $NEVENT 0 $tbbnthreads
  done
done

# echo "----------------"
# echo "running with CUPLA CPU Serial"
# echo "----------------"
# for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
# do
#   ./mainCuplaCPUSerial toyDetector_$i $DC $DELTAO $DELTAC $RHOC 1 $NEVENT 0
# done

# echo "----------------"
# echo "running with CUPLA CUDA"
# echo "----------------"
# for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
# do
#   ./mainCuplaCUDA toyDetector_$i $DC $DELTAO $DELTAC $RHOC 1 $NEVENT 0
# done


#                                 CPU [1T]     CPU TBB [16T]
# --- prepareDataStructures:      39.007 ms    29.9007 ms ( 1.3x)
# --- calculateDistanceToHigher: 174.452 ms    14.0589 ms (12.4x)
# --- calculateLocalDensity:     249.358 ms    20.4946 ms (12.1x)
# --- findSeedAndOutlier:        32.4970 ms    31.8462 ms ( 1.0x)
# --- assignClusters:             14.785 ms     4.3226 ms ( 3.4x)
