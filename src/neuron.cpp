
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#include "neuron.h"
#include "connection.h"

double Neuron::sEta = 0.05;   // overall net learning rate, [0.0..1.0]
double Neuron::sAlpha = 0.05; // momentum, multiplier of last deltaWeight, [0.0..1.0]


Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    mId = myIndex;
}

Neuron::~Neuron()
{
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double newDeltaWeight;
        newDeltaWeight = sEta * neuron.outputVal() * mGradient + sAlpha * neuron.m_outputWeights[mId].deltaWeight;
        neuron.m_outputWeights[mId].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[mId].weight += newDeltaWeight;
    }
}

double Neuron::randomWeight(void)
{
    return rand() / double(RAND_MAX);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    // derivative of weights
    double sum = 0.0;
    // TODO : check - 1
    for (int n = 0; n < (int)nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].mGradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    mGradient = dow * Neuron::transferFunctionDerivative(mOutputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - mOutputVal; // dError/dOutput (( 1/2(targetVal - mOutputVal) ^2)')
    mGradient = delta * Neuron::transferFunctionDerivative(mOutputVal);
}

double Neuron::transferFunction(double x)
{
    // TODO check if input or output
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    // TODO check if input or output
    // double thx = tanh(x);
    double res = 1.0 - x * x;
    return res;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].outputVal() * prevLayer[n].m_outputWeights[mId].weight;
    }
    mOutputVal = Neuron::transferFunction(sum);
}

double Neuron::outputVal() const
{
    return mOutputVal;
}

void Neuron::setOutputVal(double newOutputVal)
{
    mOutputVal = newOutputVal;
}

double Neuron::inputVal() const
{
    return mInputVal;
}

void Neuron::setInputVal(double newInputVal)
{
    mInputVal = newInputVal;
}

double Neuron::gradient() const
{
    return mGradient;
}

void Neuron::setGradient(double newGradient)
{
    mGradient = newGradient;
}

int Neuron::id() const
{
    return mId;
}

void Neuron::setId(int newId)
{
    mId = newId;
}
