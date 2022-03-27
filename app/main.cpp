// Executables must have the following defined if the library contains
// doctest definitions. For builds with this disabled, e.g. code shipped to
// users, this can be left out.
#ifdef ENABLE_DOCTEST_IN_LIBRARY
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#endif

#include <iostream>
#include <stdlib.h>

#include "connection.h"
#include "trainingData.h"
#include "neuron.h"
#include "net.h"
#include "utils.h"

using namespace std;

int main()
{
    cout << "Executing app in : " << endl;
    system("pwd");

    // TrainingData trainData("/home/mat/programming/c_cpp/nn_from_scratch/data/trainingData.txt");

    string base_dir = "/home/mat/programming/c_cpp/nn_from_scratch/data/";
    // string img_path = base_dir + "train-images-idx3-ubyte";
    string img_path = base_dir + "t10k-images-idx3-ubyte";
    vector<vector<double>> inputs;
    readMNIST(img_path, 10000, 784, inputs);

    displayNumber(inputs, 0);

    // string labelsPath = base_dir + "train-labels-idx1-ubyte";
    string labelsPath = base_dir + "t10k-labels-idx1-ubyte";
    vector<string> labs;
    readMNISTLabels(labelsPath, labs);
    displayNumberLabel(labs, 0);
    displayLabels(labs, 20);

    vector<unsigned> topology = {784, 196, 98, 10};

    Net nn(topology);
    vector<double> inputVals, targetVals, resultVals;
    inputVals.resize(784);
    targetVals.resize(10);
    resultVals.resize(10);

    unsigned trainingPass = 0;
    int val = 0;
    while (trainingPass < labs.size())
    {
        cout << endl;
        cout << "Pass: " << trainingPass << endl;

        // Get new input data and feed it forward:
        inputVals = inputs[trainingPass];
        nn.feedForward(inputVals);

        nn.getResults(resultVals);

        // serialize targetVals
        val = stoi(labs[trainingPass]);
        cout << "Target val: " << val << endl;
        fill(targetVals.begin(), targetVals.end(), 0);
        targetVals[val] = 1;

        nn.backProp(targetVals);

        // Report how well the training is working, average over recent samples:

        cout << "tar {";
        for (unsigned i = 0; i < targetVals.size(); i++)
        {
            cout << targetVals[i] << ", ";
        }
        cout << "}" << endl;

        cout << "res {";
        for (unsigned i = 0; i < resultVals.size(); i++)
        {
            cout << resultVals[i] << ", ";
        }
        cout << "}" << endl;

        // cout << "Net recent average error: " << nn.getRecentAverageError() << endl;
        ++trainingPass;
    }

    cout << "Done" << endl;

    return 0;

    // // e.g., { 3, 2, 1 }
    // vector<unsigned> topology;
    // trainData.getTopology(topology);

    // Net myNet(topology);

    // vector<double> inputVals, targetVals, resultVals;
    // int trainingPass = 0;

    // while (!trainData.isEof())
    // {
    //     ++trainingPass;
    //     cout << endl;
    //     cout << "Pass " << trainingPass;

    //     // Get new input data and feed it forward:
    //     if (trainData.getNextInputs(inputVals) != topology[0])
    //     {
    //         break;
    //     }
    //     showVectorVals(": Inputs:", inputVals);
    //     myNet.feedForward(inputVals);

    //     myNet.getResults(resultVals);
    //     showVectorVals("Outputs:", resultVals);

    //     // Train the net what the outputs should have been:
    //     trainData.getTargetOutputs(targetVals);
    //     showVectorVals("Targets:", targetVals);
    //     assert(targetVals.size() == topology.back());

    //     myNet.backProp(targetVals);

    //     // Report how well the training is working, average over recent samples:
    //     cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
    // }

    // cout << "Done" << endl;

    // return 0;
}
