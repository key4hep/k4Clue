#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "CLUEAlgo.h"
#ifndef USE_CUPLA
#include "CLUEAlgoGPU.h"
#else
#include "CLUEAlgoCupla.h"
#ifdef FOR_TBB
#include "tbb/task_scheduler_init.h"
#endif
#endif

#include "read_events.h"

#include "edm4hep/MCParticle.h"
#include "podio/ROOTReader.h"

void mainRun( std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight,
              std::string outputFileName,
              float dc, float rhoc, float outlierDeltaFactor,
              bool useParallel, bool verbose  ) {

  //////////////////////////////
  // run CLUE algorithm
  //////////////////////////////
  std::cout << "Start to run CLUE algorithm" << std::endl;
  if (useParallel) {
#ifndef USE_CUPLA
    std::cout << "Using CLUEAlgoGPU ... " << std::endl;
    CLUEAlgoGPU clueAlgo(dc, rhoc, outlierDeltaFactor,
			 verbose);
    clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
    // measure excution time of makeClusters
    auto start = std::chrono::high_resolution_clock::now();
    clueAlgo.makeClusters();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << " | Elapsed time: " << elapsed.count()*1000 << " ms\n";
    // output result to outputFileName. -1 means all points.
    if(verbose)
      clueAlgo.verboseResults(outputFileName, -1);
#else
    std::cout << "Using CLUEAlgoCupla ... " << std::endl;
    CLUEAlgoCupla<cupla::Acc> clueAlgo(dc, rhoc, outlierDeltaFactor,
				       verbose);
    clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
    // measure excution time of makeClusters
    auto start = std::chrono::high_resolution_clock::now();
    clueAlgo.makeClusters();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
    // output result to outputFileName. -1 means all points.
    if(verbose)
      clueAlgo.verboseResults(outputFileName, -1);
#endif
  } else {
    std::cout << "Using CLUEAlgo ... " << std::endl;
    CLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
    clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
    // measure excution time of makeClusters
    auto start = std::chrono::high_resolution_clock::now();
    clueAlgo.makeClusters();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
    // output result to outputFileName. -1 means all points.
    if(verbose)
      clueAlgo.verboseResults(outputFileName, -1);
  }

  std::cout << "Finished running CLUE algorithm" << std::endl;
} // end of testRun()



int main(int argc, char *argv[]) {

  //////////////////////////////
  // MARK -- set algorithm parameters
  //////////////////////////////
  float dc=20.f, rhoc=80.f, outlierDeltaFactor=2.f;
  bool useParallel=false;
  bool doBarrel = false;
  bool verbose = false;

  int TBBNumberOfThread = 1;

  if (argc == 8 || argc == 9) {
    dc = std::stof(argv[2]);
    rhoc = std::stof(argv[3]);
    outlierDeltaFactor = std::stof(argv[4]);
    useParallel = (std::stoi(argv[5])==1)? true:false;
    doBarrel = (std::stoi(argv[6])==1)? true:false;
    verbose = (std::stoi(argv[7])==1)? true:false;
    if (argc == 9) {
      TBBNumberOfThread = std::stoi(argv[8]);
      if (verbose) {
        std::cout << "Using " << TBBNumberOfThread;
	std::cout << " TBB Threads" << std::endl;
      }
    }
  } else {
    std::cout << "bin/main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useParallel] [doBarrel] [verbose] [NumTBBThreads]" << std::endl;
    return 1;
  }

#ifdef FOR_TBB
  if (verbose) {
    std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads" << std::endl;
  }
  tbb::task_scheduler_init init(TBBNumberOfThread);
#endif

  //////////////////////////////
  // read data 
  //////////////////////////////
  std::string inputFileName = argv[1];

  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  if(inputFileName.find(".root")!=std::string::npos){
    //TO BE FIXED: read only one event
    read_events<podio::ROOTReader>(inputFileName, x, y, layer, weight, 1, doBarrel);
    //read all events in the file
    //read_events<podio::ROOTReader>(inputFileName);
  } else if (inputFileName.find(".csv")!=std::string::npos){
    read_events_from_csv(inputFileName, x, y, layer, weight);
  } else {
    std::cerr << "Not sure how to read this input file." << std::endl;
  }

  if(x.empty() || y.empty() || layer.empty() || weight.empty()){
    std::cerr << "Data is empty. Break." << std::endl;
    return 0;
  }

  std::string underscore = "_", suffix = "";
  suffix.append(underscore);
  suffix.append(std::to_string(int(dc)));
  suffix.append(underscore);
  suffix.append(std::to_string(int(deltao)));
  suffix.append(underscore);
  suffix.append(std::to_string(int(deltac)));
  suffix.append(underscore);
  suffix.append(std::to_string(int(rhoc)));
  if(doBarrel)
    suffix.append("_Barrel");
  else
    suffix.append("_Endcap");
  suffix.append(".csv");
  size_t pos; std::string toReplace;
  if(inputFileName.find(".root")!=std::string::npos){
    toReplace = ".root";
    pos = inputFileName.find(toReplace);
  } else if (inputFileName.find(".csv")!=std::string::npos){
    toReplace = ".csv";
    pos = inputFileName.find(toReplace);
  } 
  std::string outputFileName = inputFileName.replace(pos, toReplace.length(), suffix);
  std::cout << "Output file: " << outputFileName << std::endl;

  //////////////////////////////
  // MARK -- test run
  //////////////////////////////
  mainRun(x, y, layer, weight,
          outputFileName,
          dc, rhoc, outlierDeltaFactor,
          useParallel, verbose);

  return 0;
}
