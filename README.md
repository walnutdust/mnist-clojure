# mnist-clojure

### Implementation of various machine learning algorithms for MNIST

An exploration of using Clojure for machine learning to differentiate digits in the MNIST files, inspired by Clojure's
philosophy that everything is data. 

Current implementation in this repository is a single layer perceptron that generates the probability of its predictions 
via soft-max. If the settings are kept as they are, the overall accuracy is 83.08%.

## Usage

1. Clone the repository onto your local machine
1. Open the REPL and run `train-and-test` in `src/mnist_clojure/core.clj`. A function is available in the comments at
the bottom of the code.
1. Adjust the desired parameters for experimentation!

## Reflections

Clojure as a language might allow for higher-level programming idiomatically, but these features should be only used 
when optimization is not an issue. In this case, the fact that the structures of the input data was pretty regular,
and that all of it had to be parsed, alongside the amount of information that needed to be processed should have been
indicators that a lower-level programming paradigm ought to be adopted for optimization purposes, since lazy programming
barely helps with saving computation power. There are Clojure ML Clojars out there, and it can be seen that most of 
these libraries utilizes this learning to help increase their computation power.

Nevertheless, this project served as a good exercise in applying idiomatic Clojure for the purposes of machine learning.
It was interesting to read up about the perceptron and recapturing the math and the structure into code. For purposes
of testing as well, a pseudo-random algorithm was also used, and can be replaced by the user with a random library for
better results.

## Contributors

* Garett Tok Ern Liang [(walnutdust)](https://github.com/walnutdust/)
* David Lee [(deejayessel)](https://github.com/deejayessel/)

## License

This project is licenced under the EPL v 2.0.
