#include <iostream>
#include <iomanip>
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

#include "IO_helper.h"

//EDM4HEP libraries
#include "podio/ROOTReader.h"
#include "podio/EventStore.h"

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string create_outputfileName(std::string inputFileName, float dc,
                                 float rhoc, float outlierDeltaFactor,
                                 bool useParallel,
                                 std::string eventNumber){
  std::string underscore = "_", suffix = "";
  suffix.append(underscore);
  suffix.append(to_string_with_precision(dc,2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(rhoc,2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(outlierDeltaFactor,2));
  suffix.append(underscore);
  suffix.append(eventNumber+"event");
  suffix.append(".csv");

  std::string outputFileName = inputFileName;
  replace(outputFileName, "input", "output");
  if(inputFileName.find(".root")!=std::string::npos){
    replace(outputFileName, ".root", ".csv");
  }
  replace(outputFileName, ".csv", suffix);

  return outputFileName;
}

bool emptyEvent(std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight){

  if(x.empty() || y.empty() || layer.empty() || weight.empty()){
    std::cerr << "Data is empty. Break." << std::endl;
    return true;
  }

  return false;

}

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
    CLUEAlgoCuplaT<cupla::Acc, LayerTilesConstants> clueAlgo(dc, rhoc, outlierDeltaFactor,
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
    CLUEAlgoT<LayerTilesConstants> clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
    clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
    // measure excution time of makeClusters
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "clueAlgo.makeClusters" << std::endl;
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
  bool verbose = false;

  int TBBNumberOfThread = 1;

  std::string inputFileName = argv[1];
  if (argc == 7 || argc == 8) {
    dc = std::stof(argv[2]);
    rhoc = std::stof(argv[3]);
    outlierDeltaFactor = std::stof(argv[4]);
    useParallel = (std::stoi(argv[5])==1)? true:false;
    verbose = (std::stoi(argv[6])==1)? true:false;
    if (argc == 8) {
      TBBNumberOfThread = std::stoi(argv[7]);
      if (verbose) {
        std::cout << "Using " << TBBNumberOfThread;
	std::cout << " TBB Threads" << std::endl;
      }
    }
  } else {
    std::cout << "bin/main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useParallel] [verbose] [NumTBBThreads]" << std::endl;
    return 1;
  }

#ifdef FOR_TBB
  if (verbose) {
    std::cout << "Setting up " << TBBNumberOfThread << " TBB Threads" << std::endl;
  }
  tbb::task_scheduler_init init(TBBNumberOfThread);
#endif

  //////////////////////////////
  // Read data and run algo
  //////////////////////////////
  edm4hep::CalorimeterHitCollection calo_coll;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  // Read EDM4HEP data
  if(inputFileName.find(".root")!=std::string::npos){

    std::cout<<"input edm4hep file: "<<inputFileName<<std::endl;
    podio::ROOTReader reader;
    reader.openFile(inputFileName);

    podio::EventStore store;
    store.setReader(&reader);
    std::string bitFieldCoder = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16" ;

    unsigned nEvents = reader.getEntries();
    int padding = std::to_string(nEvents).size();

    for(unsigned i=0; i<nEvents; ++i) {
      if(verbose)  std::cout<<"reading event "<<i<<std::endl;

      const auto& EB_calo_coll = store.get<edm4hep::CalorimeterHitCollection>("ECALBarrel");
      if( EB_calo_coll.isValid() ) {
        for(const auto& calo_hit_EB : EB_calo_coll){
          calo_coll->push_back(calo_hit_EB.clone());
        }
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << EB_calo_coll.size() << " caloHits in Barrel." << std::endl;

      const auto& EE_calo_coll = store.get<edm4hep::CalorimeterHitCollection>("ECALEndcap");
      if( EE_calo_coll.isValid() ) {
        for(const auto& calo_hit_EE : EE_calo_coll ){
          calo_coll->push_back(calo_hit_EE.clone());
        }
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << EE_calo_coll.size() << " caloHits in Endcap." << std::endl;
    
      std::cout << calo_coll->size() << " caloHits in total. " << std::endl;
      read_EDM4HEP_event(calo_coll, bitFieldCoder, x, y, layer, weight);

      std::string eventString = std::to_string(i);
      eventString.insert(eventString.begin(), padding - eventString.size(), '0');

      std::string outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor, useParallel, eventString);
      if( !emptyEvent(x, y, layer, weight)){
        mainRun(x, y, layer, weight,
                outputFileName,
                dc, rhoc, outlierDeltaFactor,
                useParallel, verbose);
      }
      if(verbose){
        std::cout << "Output file: " << outputFileName << std::endl;
        std::cout << std::endl;
      }

      // Cleaning
      calo_coll.clear();
      x.clear();
      y.clear();
      layer.clear();
      weight.clear();
      store.clear();
      reader.endOfEvent();
    }

    reader.closeFile();

  // Read data in csv file
  } else if (inputFileName.find(".csv")!=std::string::npos){
    read_from_csv(inputFileName, x, y, layer, weight);
    if( !emptyEvent(x, y, layer, weight)){
      std::string outputFileName = create_outputfileName(inputFileName, dc, rhoc, outlierDeltaFactor, useParallel, "0");
      mainRun(x, y, layer, weight,
              outputFileName,
              dc, rhoc, outlierDeltaFactor,
              useParallel, verbose);
      if(verbose)
        std::cout << "Output file: " << outputFileName << std::endl;
    } else {
      std::cerr << "Something wrong with the input file" << std::endl;
    }
  } else {
    std::cerr << "Not sure how to read this input file." << std::endl;
  }

  return 0;

}

