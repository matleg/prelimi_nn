
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

void showVectorVals(string label, vector<double> &v);
int reverseInt(int i);
void readMNIST(string file_path, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);
void readMNISTLabels(string file_path, vector<string> &arr);
void displayNumber(vector<vector<double>> &arr, int i);
void displayNumberLabel(vector<string> &arr, int i);
void displayLabels(vector<string> &arr, int nb);
