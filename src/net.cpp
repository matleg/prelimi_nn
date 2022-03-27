
#include <iostream>

#include "net.h"
#include "connection.h"
#include "neuron.h"

using namespace std;

double Net::mRecentAverageSmoothingFactor = 10.0; // Number of training samples to average over

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        mLayers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // we have made a new layer, now fill it with neurons and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            mLayers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        cout << "Made " << topology[layerNum] + 1 << " neurons!" << endl;
        // force the bias node's output value to 1.0. It's the last neuron created above
        mLayers.back().back().setOutputVal(1.0);
    }
    std::cout << "Made " << numLayers << " layers!" << std::endl;
}

Net::~Net()
{
}

void Net::getResults(vector<double> &resultVals)
{
    resultVals.clear();
    // TODO : check - 1
    for (unsigned n = 0; n < mLayers.back().size() - 1; ++n)
    {
        resultVals.push_back(mLayers.back()[n].outputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    Layer &outputLayer = mLayers.back();
    mError = 0.0;
    // TODO : check -1
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].outputVal();
        mError += delta * delta;
    }

    // TODO : check -1
    mError /= outputLayer.size() - 1;
    mError = sqrt(mError);
    mRecentAverageError = (mRecentAverageError * mRecentAverageSmoothingFactor + mError) / (mRecentAverageSmoothingFactor + 1.0);

    // TODO : check -1
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = mLayers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = mLayers[layerNum];
        Layer &nextLayer = mLayers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned layerNum = mLayers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = mLayers[layerNum];
        Layer &prevLayer = mLayers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == mLayers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        mLayers[0][i].setOutputVal(inputVals[i]);
    }

    for (unsigned layerNum = 1; layerNum < mLayers.size(); ++layerNum)
    {
        Layer &prevLayer = mLayers[layerNum - 1];
        for (unsigned n = 0; n < mLayers[layerNum].size() - 1; ++n)
        {
            mLayers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::startTraining()
{
    // TODO : use it!
    while (mTrainingIndex < (int)mInputs.size())
    {
        cout << "Index " << mTrainingIndex;
        vector<double> resultVals;

        feedForward(mInputs[mTrainingIndex]);

        getResults(resultVals);

        assert(mOutputs[mTrainingIndex].size() == mLayers.back().size());

        backProp(mOutputs[mTrainingIndex]);

        cout << "Net recent average error: " << getRecentAverageError() << "\n";
        ++mTrainingIndex;
    }

    cout << "Done";
}

void Net::randomizeConnectionsWeight(void)
{
    for (int l = 0; l < (int)mLayers.size() - 1; l++)
    {
        for (int n = 0; n < (int)mLayers[l].size(); n++)
        {
            mLayers[l][n].randomWeight();
        }
    }
}

void Net::setBiaisOutputVal(void)
{
    for (int l = 0; l < (int)mLayers.size() - 1; l++)
    {
        mLayers[l].back().setOutputVal(1);
    }
}
