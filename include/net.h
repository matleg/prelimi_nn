#ifndef NET_H
#define NET_H

#include <vector>
#include <iostream>

#include "neuron.h"

class Net
{
public:
    Net(const vector<unsigned> &topology);
    ~Net();
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals);
    double getRecentAverageError(void) const { return mRecentAverageError; }

    void randomizeConnectionsWeight(void); // not used
    void setBiaisOutputVal(void); // not used

    void startTraining();


private:
    vector<Layer> mLayers;
    double mError;
    double mRecentAverageError;
    static double mRecentAverageSmoothingFactor;
    vector<vector<double>> mInputs;
    vector<vector<double>> mOutputs;

    int mTrainingIndex;
};

#endif // NET_H
