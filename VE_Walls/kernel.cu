#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thrust/complex.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const int MAX_THREADS = 1024;

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        cout << "CUDA Runtime Error: " << cudaGetErrorName(result) << " - " << cudaGetErrorString(result) << endl;
    }

    return result;
}

__global__ void radiusKernel(double *inputs, int pointCount, double neighborRadius, int *radiusNeighborCount) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pointCount) return;

    int i;

    double x = inputs[3 * index];
    double y = inputs[3 * index + 1];
    double z = inputs[3 * index + 2];

    double neighborX;
    double neighborY;
    double neighborZ;
    double neighborDistance;

    for (i = 0; i < pointCount; i++) {
        if (index == i) continue;

        neighborX = inputs[3 * i];
        neighborY = inputs[3 * i + 1];
        neighborZ = inputs[3 * i + 2];

        neighborDistance = sqrtf(
            (x - neighborX) * (x - neighborX) +
            (y - neighborY) * (y - neighborY) +
            (z - neighborZ) * (z - neighborZ)
        );

        if (neighborDistance <= neighborRadius) {
            radiusNeighborCount[index]++;
        }
    }
}

__global__ void kernel(
    double *inputs, double *features, double *bestNeighbors, double *bestNeighborsIndeces, double *OO,
    int pointCount, int vicinityAlgo, int neighborCount, double neighborRadius,
    double *radiusBestNeighborsIndeces, double *radiusOO, int *radiusNeighborCount
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pointCount) return;

    int i, j, k;

    double x = inputs[3 * index];
    double y = inputs[3 * index + 1];
    double z = inputs[3 * index + 2];

    double neighborX;
    double neighborY;
    double neighborZ;
    double neighborDistance;

    double mi[3];
    mi[0] = 0.0; mi[1] = 0.0; mi[2] = 0.0;

    float S[3][3];

    if (vicinityAlgo == 1) {
        for (i = 0; i < neighborCount; i++) {
            bestNeighbors[index * neighborCount + i] = INFINITY;
            bestNeighborsIndeces[index * neighborCount + i] = INFINITY;
        }

        for (i = 0; i < pointCount; i++) {
            if (index == i) continue;

            neighborX = inputs[3 * i];
            neighborY = inputs[3 * i + 1];
            neighborZ = inputs[3 * i + 2];

            neighborDistance = sqrtf(
                (x - neighborX) * (x - neighborX) +
                (y - neighborY) * (y - neighborY) +
                (z - neighborZ) * (z - neighborZ)
            );

            //if (index == 0) printf("%g ", neighborDistance);

            for (j = 0; j < neighborCount; j++) {
                if (neighborDistance < bestNeighbors[index * neighborCount + j]) {
                    for (k = neighborCount - 1; k > j; k--) {
                        bestNeighbors[index * neighborCount + k] = bestNeighbors[index * neighborCount + k - 1];
                        bestNeighborsIndeces[index * neighborCount + k] = bestNeighborsIndeces[index * neighborCount + k - 1];
                    }

                    bestNeighbors[index * neighborCount + j] = neighborDistance;
                    bestNeighborsIndeces[index * neighborCount + j] = i;

                    break;
                }
            }
        }

        for (i = 0; i < neighborCount; i++) {
            mi[0] += inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 0] / neighborCount;
            mi[1] += inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 1] / neighborCount;
            mi[2] += inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 2] / neighborCount;
            //if (index == 0) printf("%f\t%f\t%f\n", mi[0], mi[1], mi[2]);
        }

        for (i = 0; i < neighborCount; i++) {
            OO[index * neighborCount * 3 + (i * 3 + 0)] = inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 0] - mi[0];
            OO[index * neighborCount * 3 + (i * 3 + 1)] = inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 1] - mi[1];
            OO[index * neighborCount * 3 + (i * 3 + 2)] = inputs[3 * (int)bestNeighborsIndeces[index * neighborCount + i] + 2] - mi[2];
            //if (index == 2000) printf("%d: %f %f %f\n", i, OO[i * 3 + 0], OO[i * 3 + 1], OO[i * 3 + 2]);
        }

        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                S[i][j] = 0.0;
                for (k = 0; k < neighborCount; k++) {
                    S[i][j] += OO[index * neighborCount * 3 + (k * 3 + j)] * OO[index * neighborCount * 3 + (k * 3 + i)];
                }
                S[i][j] /= neighborCount;
                //if (index == 2000) printf("%.17g ", S[i][j]);
            }
            //if (index == 2000) printf("\n");
        }
    }
    else if (vicinityAlgo == 2) {
        int previousIndecesSum = 0;

        if (radiusNeighborCount[index] == 0) {
            return;
        }

        for (j = 0; j < index; j++) {
            previousIndecesSum += radiusNeighborCount[j];
            //if (index == 2) printf("%d: %d\n", index, previousIndecesSum);
        }

        int neighborsFoundCount = 0;

        for (i = 0; i < pointCount; i++) {
            if (index == i) continue;

            neighborX = inputs[3 * i];
            neighborY = inputs[3 * i + 1];
            neighborZ = inputs[3 * i + 2];

            neighborDistance = sqrtf(
                (x - neighborX) * (x - neighborX) +
                (y - neighborY) * (y - neighborY) +
                (z - neighborZ) * (z - neighborZ)
            );

            if (neighborDistance <= neighborRadius) {
                radiusBestNeighborsIndeces[previousIndecesSum + neighborsFoundCount] = i;
                neighborsFoundCount++;
            }
        }
        
        for (i = 0; i < radiusNeighborCount[index]; i++) {
            mi[0] += inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 0] / radiusNeighborCount[index];
            mi[1] += inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 1] / radiusNeighborCount[index];
            mi[2] += inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 2] / radiusNeighborCount[index];
            //if (index == 2000) printf("%f\t%f\t%f\n", mi[0], mi[1], mi[2]);
        }
        
        for (i = 0; i < radiusNeighborCount[index]; i++) {
            radiusOO[3 * previousIndecesSum + i + 0 * radiusNeighborCount[index]] = 
                inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 0] - mi[0];

            radiusOO[3 * previousIndecesSum + i + 1 * radiusNeighborCount[index]] = 
                inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 1] - mi[1];

            radiusOO[3 * previousIndecesSum + i + 2 * radiusNeighborCount[index]] = 
                inputs[3 * (int)radiusBestNeighborsIndeces[previousIndecesSum + i] + 2] - mi[2];
            //if (index == 2000) printf("%d: %g %f %f\n", i, 
            //    radiusOO[3 * previousIndecesSum + i + 0 * radiusNeighborCount[index]],
            //    radiusOO[3 * previousIndecesSum + i + 1 * radiusNeighborCount[index]],
            //    radiusOO[3 * previousIndecesSum + i + 2 * radiusNeighborCount[index]]);
        }

        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                S[i][j] = 0.0;

                for (k = 0; k < radiusNeighborCount[index]; k++) {
                    S[i][j] += 
                        radiusOO[3 * previousIndecesSum + i * radiusNeighborCount[index] + k] * 
                        radiusOO[3 * previousIndecesSum + j * radiusNeighborCount[index] + k];
                }

                S[i][j] /= radiusNeighborCount[index];
            }
        }
    }

    thrust::complex<double> im = thrust::complex<double>(0.0, 1.0f);

    double a = S[0][0];
    double b = S[1][1];
    double c = S[2][2];
    double d = S[0][1];
    double e = S[0][2];
    double f = S[1][2];

    double lambda1, lambda2, lambda3;

    thrust::complex<double> lambdaPart1 = thrust::pow(2 * a*a*a - 3 * a*a*b - 3 * a*a*c +
        thrust::sqrt(thrust::complex<double>(4) * thrust::pow(thrust::complex<double>(-a*a + a*b + a*c - b*b + b*c - c*c - 3 * d*d - 3 * e*e - 3 * f*f, 0.0), thrust::complex<double>(3)) +
        thrust::pow(thrust::complex<double>(2 * a*a*a - 3 * a*a*b - 3 * a*a*c - 3 * a*b*b + 12 * a*b*c - 3 * a*c*c + 9 * a*d*d + 9 * a*e*e -
        18 * a*f*f + 2 * b*b*b - 3 * b*b*c - 3 * b*c*c + 9 * b*d*d - 18 * b*e*e +
        9 * b*f*f + 2 * c*c*c - 18 * c*d*d + 9 * c*e*e + 9 * c*f*f + 54 * d*e*f), thrust::complex<double>(2))) -
        3 * a*b*b + 12 * a*b*c - 3 * a*c*c + 9 * a*d*d + 9 * a*e*e - 18 * a*f*f + 2 * b*b*b -
        3 * b*b*c - 3 * b*c*c + 9 * b*d*d - 18 * b*e*e + 9 * b*f*f + 2 * c*c*c -
        18 * c*d*d + 9 * c*e*e + 9 * c*f*f + 54 * d*e*f, thrust::complex<double>(1 / 3.0));

    thrust::complex<double> lambdaPart2 = -a*a + a*b + a*c - b*b + b*c - c*c - 3 * d*d - 3 * e*e - 3 * f*f;

    lambda1 = (1/(3*cbrt(2.0)) * lambdaPart1 -
        cbrt(2.0) * lambdaPart2 /
        (3.0 * lambdaPart1) + 
        (a+b+c)/3.0)
        .real()
    ;

    lambda2 = ((-(1.0 + im * sqrt(3.0)) / (6.0 * cbrt(2.0))) * lambdaPart1 +
        (1.0 - im * sqrt(3.0)) * lambdaPart2 /
        (3.0 * thrust::pow(thrust::complex<double>(2), 2 / 3.0) * lambdaPart1) +
        (a + b + c) / 3.0)
        .real()
    ;

    lambda3 = ((-(1.0 - im * sqrt(3.0))/(6.0 * cbrt(2.0))) * lambdaPart1 +
        (1.0 + im * sqrt(3.0)) * lambdaPart2 /
        (3.0 * thrust::pow(thrust::complex<double>(2), 2/3.0) * lambdaPart1) +
        (a + b + c) / 3.0)
        .real()
    ;
    
    if (index == 32) printf("%d: %.17g\n", index, lambda1);
    if (index == 32) printf("%.17g\n", lambda2);
    if (index == 32) printf("%.17g\n", lambda3);

    if (index == 0) printf("\n");

    features[6 * index + 0] = (lambda1 - lambda2) / lambda1;
    features[6 * index + 1] = (lambda2 - lambda3) / lambda1;
    features[6 * index + 2] = lambda3 / lambda1;
    features[6 * index + 3] = cbrt(lambda1 * lambda2 * lambda3);
    features[6 * index + 4] = (lambda1 - lambda3) / lambda1;
    features[6 * index + 5] = -((lambda1 * log(lambda1)) + (lambda2 * log(lambda2)) + (lambda3 * log(lambda3)));

    // TODO: check if the lambda order is correct, i.e. l1 >= l2 >= l3

    // TODO: check feature values (sometimes eigenentropy is -nan(ind))
}

int main(int argc, char* argv[])
{
    int i, j;

    string inputName;
    string outputName;
    ifstream inputFile;
    ofstream outputFile;

    int vicinityAlgo = 0; // 1: kNN, 2: FDN
    int neighborCount = 0;
    double neighborRadius = 0.0;

    for (i = 1; i < argc; ++i) {
        if (i + 1 < argc) {
            if (string(argv[i]) == "-i") {
                inputName = argv[++i];
            }
            if (string(argv[i]) == "-o") {
                outputName = argv[++i];
            }
            if (string(argv[i]) == "-n") {
                neighborCount = stoi(argv[++i]);

                vicinityAlgo += 1;
            }
            if (string(argv[i]) == "-r") {
                neighborRadius = stod(argv[++i]);

                vicinityAlgo += 2;
            }
        }
    }

    if (vicinityAlgo == 0) {
        cout << "ERROR: No vicinity algorithm parameters specified!" << endl;
    }
    else if (vicinityAlgo == 3) {
        cout << "ERROR: Too many vicinity algorithm parameters specified!" << endl;
    }
    else if (neighborCount <= 0 && neighborRadius <= 0.0) {
        cout << "ERROR: Incorrect vicinity algorithm parameters!" << endl;
    }
    else {
        cout << "Neighbor count : " << neighborCount << endl;
        cout << "Neighbor radius: " << neighborRadius << endl;

        inputFile.open(inputName);
        outputFile.open(outputName);

        unsigned long long int inputFileLineNumber = (int)count(
            istreambuf_iterator<char>(inputFile),
            istreambuf_iterator<char>(),
            '\n'
        );
        inputFile.seekg(0);

        cout << "Points count: " << inputFileLineNumber << endl;

        string inputLine;

        double *inputs = (double*)malloc(3 * inputFileLineNumber * sizeof(double));

        int lineCounter = 0;

        while (getline(inputFile, inputLine)) {
            stringstream stream(inputLine);

            string s;

            for (i = 0; i < 3; i++) {
                getline(stream, s, ' ');
                inputs[lineCounter * 3 + i] = stof(s);
                //cout << inputs[lineCounter * 3 + i] << " ";
            }
            //cout << endl;

            lineCounter++;
        }

        checkCuda(cudaDeviceReset());
        checkCuda(cudaSetDevice(0));

        double *cudaInputs;

        checkCuda(cudaMalloc((double**)&cudaInputs, 3 * inputFileLineNumber * sizeof(double)));

        checkCuda(cudaMemcpy(cudaInputs, inputs, 3 * inputFileLineNumber * sizeof(double), cudaMemcpyHostToDevice));

        int *cudaRadiusNeighborCount;

        checkCuda(cudaMalloc((int**)&cudaRadiusNeighborCount, inputFileLineNumber * sizeof(int)));

        int *radiusNeighborCount = (int*)calloc(inputFileLineNumber, sizeof(int));
        int radiusNeighborCountTotal = 0;

        if (vicinityAlgo == 2) {
            dim3 threadsPerBlock(inputFileLineNumber);
            dim3 blocksPerGrid(1);

            if (inputFileLineNumber > MAX_THREADS) {
                int divisor = (int)ceil((float)inputFileLineNumber / MAX_THREADS);
                threadsPerBlock.x = (int)ceil(1.0 * inputFileLineNumber / divisor);
                blocksPerGrid.x = divisor;
            }

            radiusKernel <<< blocksPerGrid, threadsPerBlock >>> (cudaInputs, inputFileLineNumber, neighborRadius, cudaRadiusNeighborCount);

            checkCuda(cudaMemcpy(radiusNeighborCount, cudaRadiusNeighborCount, inputFileLineNumber * sizeof(int), cudaMemcpyDeviceToHost));

            //for (i = 0; i < inputFileLineNumber; i++) {
                //cout << i << ": " << radiusNeighborCount[i] << endl;
            //}

            radiusNeighborCountTotal = accumulate(radiusNeighborCount, radiusNeighborCount + inputFileLineNumber, 0);
            cout << "Total neighbors: " << radiusNeighborCountTotal << endl;
        }

        long double potentialMemory = (
            3 * inputFileLineNumber / 1024.0 / 1024.0 * sizeof(double)
            + 6 * inputFileLineNumber / 1024.0 / 1024.0 * sizeof(double)
            + (vicinityAlgo == 1 ? 1 : 0) * neighborCount / 1024.0 * inputFileLineNumber / 1024.0 * sizeof(double)
            + (vicinityAlgo == 1 ? 1 : 0) * neighborCount / 1024.0 * inputFileLineNumber / 1024.0 * sizeof(double)
            + (vicinityAlgo == 1 ? 1 : 0) * 3 * neighborCount / 1024.0 * inputFileLineNumber / 1024.0 * sizeof(double)
            + (vicinityAlgo == 2 ? 1 : 0) * radiusNeighborCountTotal / 1024.0 / 1024.0 * sizeof(double)
            + (vicinityAlgo == 2 ? 1 : 0) * 3 * radiusNeighborCountTotal / 1024.0 / 1024.0 * sizeof(double)
        );

        cout << "potentialMemory: " << potentialMemory << " MB" << endl;
        
        double *features = (double*)malloc(6 * inputFileLineNumber * sizeof(double));

        double *cudaFeatures;
        double *cudaBestNeighbors;
        double *cudaBestNeighborsIndeces;
        double *cudaOO;
        double *cudaRadiusBestNeighborsIndeces;
        double *cudaRadiusOO;
        
        checkCuda(cudaMalloc((double**)&cudaFeatures, 6 * inputFileLineNumber * sizeof(double)));
        
        checkCuda(cudaMalloc((double**)&cudaBestNeighbors, neighborCount * inputFileLineNumber * sizeof(double)));
        checkCuda(cudaMalloc((double**)&cudaBestNeighborsIndeces, neighborCount * inputFileLineNumber * sizeof(double)));
        checkCuda(cudaMalloc((double**)&cudaOO, 3 * neighborCount * inputFileLineNumber * sizeof(double)));
        
        checkCuda(cudaMalloc((double**)&cudaRadiusBestNeighborsIndeces, radiusNeighborCountTotal * sizeof(double)));
        checkCuda(cudaMalloc((double**)&cudaRadiusOO, 3 * radiusNeighborCountTotal * sizeof(double)));

        dim3 threadsPerBlock(inputFileLineNumber);
        dim3 blocksPerGrid(1);

        if (inputFileLineNumber > MAX_THREADS) {
            int divisor = (int)ceil((float)inputFileLineNumber / MAX_THREADS);
            threadsPerBlock.x = (int)ceil(1.0 * inputFileLineNumber / divisor);
            blocksPerGrid.x = divisor;
        }

        //cout << "threads x: " << threadsPerBlock.x << endl;
        //cout << "blocks  x: " << blocksPerGrid.x << endl;

        kernel <<<blocksPerGrid, threadsPerBlock >>> (
            cudaInputs, cudaFeatures, cudaBestNeighbors, cudaBestNeighborsIndeces, cudaOO,
            inputFileLineNumber, vicinityAlgo, neighborCount, neighborRadius,
            cudaRadiusBestNeighborsIndeces, cudaRadiusOO, cudaRadiusNeighborCount
        );

        checkCuda(cudaPeekAtLastError());

        checkCuda(cudaMemcpy(features, cudaFeatures, 6 * inputFileLineNumber * sizeof(double), cudaMemcpyDeviceToHost));

        for (i = 0; i < inputFileLineNumber; i++) {
            for (j = 0; j < 3; j++) {
                outputFile << inputs[i * 3 + j] << "\t";
            }
            for (j = 0; j < 6; j++) {
                outputFile << features[i * 6 + j] << (j == 5 ? "" : "\t");
                //cout << features[i * 6 + j] << "\t";
            }
            outputFile << endl;
            //cout << endl;
        }

        cudaFree(cudaFeatures);
        cudaFree(cudaInputs);
        
        free(features);
        free(inputs);
    }

    cout << endl << "DONE";
    cin.ignore();

    return 0;
}
