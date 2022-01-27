#ifndef Points_h
#define Points_h

struct Points {
  
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;
  
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])

  int n;

  void clear() {
    x.clear();
    y.clear();
    layer.clear();
    weight.clear();

    rho.clear();
    delta.clear();
    nearestHigher.clear();
    clusterIndex.clear();
    followers.clear();
    isSeed.clear();

    n = 0;
  }
};
#endif
