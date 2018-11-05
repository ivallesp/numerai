# NumerAI competing system
This repository contains a sketch of a project which goal is to **autonomously paricipate in the weekly NumerAI competitions**. It was developed in 2017 and it may require some adjustments, since the platform dynamics changed since then. The idea behind it is to build an autonomous system that trains and combines different models and keeps the best solution in the platform so that the best position in the leaderboard is always achieved. 

## Introduction
There are several important components in this project
- **Machine learning engine**: it trains a battery of machine learning models (mainly from scikit-learn) and tests the results against the leaderboard.
- **NumerAPI**: it is responsible for communicating with the Numerai platform. Typical tasks are: retrieve leaderboard status, upload a submission, download competition data, etc.
- **Logger**: its goal is to log all the operations performed by the code, for tracking and debuging purposes.
- **Tests**: used for assuring code accuracy

## Getting started
Follow the subsequent steps to have this system working in your computer
1. Clone the repository
2. Create a `settings.json` file following the strucure of `settings_template.json`
3. Read the NumerAPI instructions located under the `README.md` file in the NumerAPI folder of the root of this repository and make sure it is configured
4. Open an `ipython` REPL and run the following command: `%run src/main.py`
5. Check the log file to see what the algorithm is doing at each time

## Methods
The algorithms that have been currently implemented are the following ones
- GLMNet
- ExtraTrees 
- GLM
- Multi Layer Perceptron
- k-Nearest Neighbors 
- Support Vector Machines
- Random Forest

## Contribution
All contributions are welcome. In the following lines, a set of potential next steps
- Update the library so that it works with the current NumerAI version
- Implement the NumerAI acceptance metrics 
- Extend the model battery
- Write better tests
- Implement a stacker to combine the models

## License
...
