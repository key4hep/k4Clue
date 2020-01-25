# ./main [fileName] [dc] [deltao] [deltac] [rhoc] [useGPU] [totalNumberOfEvent] [verbose]
./main circles_1000 20 20 90 12 0 10 1
./main moons_1000 20 20 150 15 0 10 1
./main aniso_1000 20 20 50 50 0 10 1
./main sissa_1000 20 20 80 25 0 10 1
./main sissa_4000 20 10 80 80 0 10 1
./main sissa_4000 20 20 80 80 0 10 1
./main sissa_4000 20 30 80 80 0 10 1
./main sissa_4000 20 40 80 80 0 10 1
./main sissa_4000 20 60 80 80 0 10 1



export DC=5
export DELTAO=10
export DELTAC=5
export RHOC=8
./main toyDetector_1000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_2000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_3000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_4000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_5000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_6000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_7000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_8000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_9000 $DC $DELTAO $DELTAC $RHOC 0 10 1
./main toyDetector_10000 $DC $DELTAO $DELTAC $RHOC 0 10 1