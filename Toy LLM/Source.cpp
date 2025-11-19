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
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            std::string input;
            std::cout << "Input a message to the model:\n";
            std::getline(std::cin, input);

            std::cout << "How many tokens (characters, including spaces) should the model predict?\n";
            int tokenCount;
            while (!(std::cin >> tokenCount)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Enter a valid number:\n";
            }
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            std::unordered_map<std::string, int> dictionary = t.read_dict();
            if (dictionary.empty())
                throw std::runtime_error("Dictionary is empty — build it first from training data!");
            int vocab_size = (int)dictionary.size();
            int unkToken = dictionary["<unk>"];

            std::vector<std::vector<float>> embeddings = read2DVector("../embeddings.txt", embedding_dim);
            std::vector<std::vector<std::vector<float>>> weights = read3DVector("../weights.txt", embedding_dim);
            if (embeddings.empty() || weights.empty())
                throw std::runtime_error("No existing embeddings or weights found.");
            if (weights.size() < 4) {
                weights.resize(4);
                weights[3] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
            }

            std::vector<int> tokenSequence;
            for (char c : input) {
                std::string s(1, c);
                if (dictionary.count(s))
                    tokenSequence.push_back(dictionary[s]);
                else
                    tokenSequence.push_back(unkToken);
            }

            std::vector<std::string> invDict(vocab_size);
            for (auto& p : dictionary)
                invDict[p.second] = p.first;

            const int maxContext = 128;
            for (int step = 0; step < tokenCount; step++) {

                int startIdx = std::max(0, (int)tokenSequence.size() - maxContext);
                std::vector<int> contextTokens(tokenSequence.begin() + startIdx, tokenSequence.end());
                int seqLen = contextTokens.size();

                std::vector<std::vector<float>> vectorSeq(seqLen);
                for (int i = 0; i < seqLen; i++) {
                    int token = contextTokens[i];
                    if (token < 0 || token >= embeddings.size()) token = unkToken;
                    vectorSeq[i] = embeddings[token];
                }

                std::vector<std::vector<float>> Q = matMul(vectorSeq, weights[0]);
                std::vector<std::vector<float>> K = matMul(vectorSeq, weights[1]);
                std::vector<std::vector<float>> V = matMul(vectorSeq, weights[2]);

                std::vector<std::vector<float>> scores = matMul(Q, transpose(K));
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        scores[i][j] /= sqrt(embedding_dim);
                for (int i = 0; i < seqLen; i++)
                    for (int j = i + 1; j < seqLen; j++)
                        scores[i][j] = -1e9f;

                std::vector<std::vector<float>> att(seqLen);
                for (int i = 0; i < seqLen; i++)
                    att[i] = softmax(scores[i]);

                std::vector<std::vector<float>> context = matMul(att, V);
                std::vector<std::vector<float>> hidden = matAdd(context, vectorSeq);
                std::vector<std::vector<float>> logits = matMul(hidden, weights[3]);

                std::vector<float> probs = softmax(logits.back());
                probs[unkToken] = 0.0f;
                float sumProb = std::accumulate(probs.begin(), probs.end(), 0.0f);
                for (float& p : probs) p /= sumProb;

                std::random_device rd;

                std::mt19937 gen(rd());
                float temperature = 1.0f;

                std::vector<float> adjustedProbs = probs;
                float sum = 0.0f;
                for (float& p : adjustedProbs) {
                    p = std::pow(p, 1.0f / temperature);
                    sum += p;
                }
                for (float& p : adjustedProbs) p /= sum;

                std::discrete_distribution<int> dist(adjustedProbs.begin(), adjustedProbs.end());
                int nextToken = dist(gen);

                tokenSequence.push_back(nextToken);
                std::cout << invDict[nextToken];


                tokenSequence.push_back(nextToken);
                std::cout << invDict[nextToken];
            }
            std::cout << "\n";
        }


	}

	return 0;
}