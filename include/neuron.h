#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <random>
#include <map>

#include "connection.h"

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    ~Neuron();

    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    double transferFunction(double x);
    double transferFunctionDerivative(double x);

    double sumDOW(const Layer &nextLayer) const;

    double outputVal() const;
    void setOutputVal(double newOutputVal);

    double inputVal() const;
    void setInputVal(double newInputVal);

    double gradient() const;
    void setGradient(double newGradient);

    int id() const;
    void setId(int newId);

    double randomWeight(void);

private:
    static double sEta;   // [0.0..1.0] overall net training rate
    static double sAlpha; // [0.0..n] multiplier of last weight change (momentum)
                          // static int sNeuronId;

    int mId;
    double mOutputVal;
    double mInputVal;
    vector<Connection> m_outputWeights;
    double mGradient;
};

#endif // NEURON_H
