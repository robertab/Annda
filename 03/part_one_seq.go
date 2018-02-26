package main
import (
	"fmt"
	"math/rand"
// 	"sort"
	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix){
    fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
    fmt.Printf("%v\n",fa)
}

type Neuron struct {
    weights []float64
}
type Layer struct{
   num_units []Neurons

}

func InitNeuron(neuron *Neuron, dimension int){
    neuron.weights = make([]float64,dimension)
    // init random weights
    for index, _ := range neuron.weights {
    // init random threshold weight
        neuron.Weights[index] = rand.NormFloat64()
    }
}
