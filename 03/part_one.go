package main
import (
	"fmt"
// 	"math/rand"
// 	"sort"
	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix){
    fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
    fmt.Printf("%v\n",fa)
}

//
//Input: None
//Output: X is the original input pattern.
//        X_distorted is a distorted version of X.

func createData() (*mat.Dense, *mat.Dense) {
    x1 := []float64{-1, -1, 1, -1, 1, -1, -1, 1,
                    -1, -1, -1, -1, -1, 1, -1, -1,
                    -1, 1, 1, -1, -1, 1, -1, 1}
    x2 := []float64{1, -1, 1, -1, 1, -1, -1, 1,
                    1, 1, -1, -1, -1, 1, -1, -1,
                    1, 1, 1, -1, 1, 1, -1, 1}
    X := mat.NewDense(3,8, x1)
    X_dist := mat.NewDense(3,8, x2)
//     X = Sgn(X)
//     X_dist = Sgn(X_dist)
    return X,X_dist
}

//
//Input: matrix of input patterns
//Output: matrix of input patterns where all values >=0 has value +1 and 
//       values <0 has value -1

func Sgn( X *mat.Dense) (*mat.Dense) {
    for i:=0; i<3; i++{
        for j:=0; j<8; j++{
            switch {
            case X.At(i,j) < 0:
                X.Set(i,j,-1.0)
            case X.At(i,j) > 0:
                X.Set(i,j,+1.0)
            case X.At(i,j) == 0:
                X.Set(i,j,+1)
            }
        }
    }
    return X
}


//Input: matrix of input patterns
//Output: matrix of weights with zero diagonal.

func createWeights(X *mat.Dense) *(mat.Dense){
    W := mat.NewDense(8,8,nil)
    W.Product(X.T(),X)
    // Hopfield net has zero diag.
    for i:=0; i<8; i++{
        for j:=0; j<8; j++{
            if (i==j){
                W.Set(i,j,0) 
            }
        }
    }
    return W
}

func getEnergy(X_distorted *mat.Dense, W *mat.Dense){
    H:=mat.Dot(X_distorted.RowView(0),W.ColView(0))
    println(H)
}

func trainHopfieldBatch(X *mat.Dense, X_distorted *mat.Dense, W *mat.Dense ,epochs int) {
    diff := mat.NewDense(3,8, nil)
    for i:=0; i<epochs; i++{
        X_distorted.Product(X_distorted,W)
        X_distorted = Sgn(X_distorted)
        print("Epochs: ")
        println(i)
        println("trained X")
        matPrint(X_distorted)
        diff.Sub(X,X_distorted)
        println("diff")
        matPrint(diff)
        println()
        getEnergy(X_distorted,W)
    }
}

func trainHopfieldSequential(X *mat.Dense, X_distorted *mat.Dense, W *mat.Dense ,epochs int) {
//     diff := mat.NewDense(3,8, nil)
    for i:=0; i<epochs; i++{
        for p:=0; p<3; p++{
//             matPrint(X.RowView(0))
//             X_distorted.Product(X_distorted,W)
//             X_distorted = Sgn(X_distorted)
//             print("Epochs: ")
//             println(i)
//             println("trained X")
//             matPrint(X_distorted)
//             diff.Sub(X,X_distorted)
//             println("diff")
//             matPrint(diff)
//             println()
        }
    }
}
func Part_one_hopfield(){
    X,X_distorted := createData()
    W := createWeights(X)
    println("Original X")
    matPrint(X)
    println("Distorted X")
    matPrint(X_distorted)
    println("Weights")
    matPrint(W)
    epochs := 10
    trainHopfieldBatch(X,X_distorted,W,epochs)
    trainHopfieldSequential(X,X_distorted,W,epochs)
//     v := mat.NewDiagonal(8, *W)
//     matPrint(v)
}

