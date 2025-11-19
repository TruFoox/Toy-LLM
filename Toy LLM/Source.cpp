#define NOMINMAX
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include "IO.h"
#include "doMath.h"
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <windows.h>
#include <numeric>
#include <algorithm>
#include <random> 

using namespace std; // Not good practice but bite me lol


int main() {
	while (true) {
		int choice;

		cout << "Please input what you want to do:\n1. Build Model Dictionary\n2. Build Model Weights\n3. Chat with Model\n";
		cin >> choice;

		if (cin.fail() || (choice != 1 && choice != 2 && choice != 3)) { // Error handling for invalid input
			cerr << "Input error!" << endl;
			return 1;
		}

		if (choice == 1) {
			;
			training t;

			t.buildDictionary();

		}
		else if (choice == 2) {
			training t;

			t.buildWeights();

		}
        else if (choice == 3) {
            training t;

            int embedding_dim = 256;
            string in;

            cout << "Input a message to the model:" << endl;
            std::getline(cin, in);

            int numPredict;
            cout << "How many tokens (characters, including spaces) should the model predict? ";
            cin >> numPredict;
            cin.ignore();

            // Load dictionary
            std::unordered_map<std::string, int> dictionary = t.read_dict();
            if (dictionary.empty())
                throw std::runtime_error("Dictionary is empty — build it first from training data!");

            int vocab_size = static_cast<int>(dictionary.size());

            // Load embeddings and weights
            std::vector<std::vector<float>> embeddings = read2DVector("../embeddings.txt", embedding_dim);
            std::vector<std::vector<std::vector<float>>> weights = read3DVector("../weights.txt", embedding_dim);

            if (embeddings.empty() || weights.empty())
                throw std::runtime_error("No existing embeddings or weights found.");

            if (weights.size() < 4) {
                weights.resize(4, std::vector<std::vector<float>>(embedding_dim, std::vector<float>(embedding_dim, 0.0f)));
                weights[3] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
            }
            std::vector<int> tokenSequence;
            tokenSequence.reserve(in.size());
            for (char c : in) {
                std::string s(1, c);
                if (dictionary.count(s))
                    tokenSequence.push_back(dictionary[s]);
                else
                    tokenSequence.push_back(dictionary["<unk>"]);
            }
            std::vector<std::string> invDict(vocab_size);
            for (auto& p : dictionary)
                invDict[p.second] = p.first;

            for (int step = 0; step < numPredict; step++) {

                int sequenceLength = tokenSequence.size();

                // Build vector sequence
                std::vector<std::vector<float>> vectorSequence(sequenceLength);
                for (int i = 0; i < sequenceLength; i++)
                    vectorSequence[i] = embeddings[tokenSequence[i]];

                std::vector<std::vector<float>> Q = matMul(vectorSequence, weights[0]);
                std::vector<std::vector<float>> K = matMul(vectorSequence, weights[1]);
                std::vector<std::vector<float>> V = matMul(vectorSequence, weights[2]);

                std::vector<std::vector<float>> attentionScores = matMul(Q, transpose(K));
                for (int m = 0; m < sequenceLength; ++m)
                    for (int n = 0; n < sequenceLength; ++n)
                        attentionScores[m][n] /= sqrt(embedding_dim);

                for (int m = 0; m < sequenceLength; ++m)
                    for (int n = m + 1; n < sequenceLength; ++n)
                        attentionScores[m][n] = -1e9f;

                std::vector<std::vector<float>> attentionWeights(sequenceLength);
                for (int m = 0; m < sequenceLength; ++m)
                    attentionWeights[m] = softmax(attentionScores[m]);

                std::vector<std::vector<float>> context = matMul(attentionWeights, V);
                std::vector<std::vector<float>> hidden = matAdd(context, vectorSequence);
                std::vector<std::vector<float>> output = matMul(hidden, weights[3]);

                std::vector<float> lastProb = softmax(output.back());

                int nextToken = 0;
                float maxVal = lastProb[0];
                for (int i = 1; i < vocab_size; i++) {
                    if (lastProb[i] > maxVal) {
                        maxVal = lastProb[i];
                        nextToken = i;
                    }
                }

                // Append token to sequence
                tokenSequence.push_back(nextToken);

                cout << invDict[nextToken];
            }

            cout << endl;

        }
	}

	return 0;
}