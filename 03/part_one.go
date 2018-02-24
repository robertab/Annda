package main
import (
	"fmt"
// 	"math"
// 	"sort"
	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix){
    fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
    fmt.Printf("%v\n",fa)
}

func createData() (*mat.Dense, *mat.Dense) {
    x1 := []float64{-1, -1, 1, -1, 1, -1, -1, 1,
                    -1, -1, -1, -1, -1, 1, -1, -1,
                    -1, 1, 1, -1, -1, 1, -1, 1}
    x2 := []float64{1, 0, 1, 0, 1, 0, 0, 1,
                    1, 1, 0, 0, 0, 1, 0, 0,
                    1, 1, 1, 0, 1, 1, 0, 1}
    X := mat.NewDense(3,8, x1)
    X_dist := mat.NewDense(3,8, x2)
    X = Sgn(X)
    X_dist = Sgn(X_dist)
    return X,X_dist
}

func Sgn( X *mat.Dense) (*mat.Dense) {
    for i:=0; i<3; i++{
        for j:=0; j<8; j++{
            switch {
            case X.At(i,j) < 0:
                X.Set(i,j,-1.0)
            case X.At(i,j) > 0:
                X.Set(i,j,+1.0)
            case X.At(i,j) == 0:
                X.Set(i,j,-1)
            }
        }
    }
    return X
}
func createWeights(X *mat.Dense) *(mat.Dense){
    W := mat.NewDense(8,8,nil)
    W.Product(X.T(),X)
    return W
}

func trainHopfieldBatch(X *mat.Dense, X_distorted *mat.Dense, W *mat.Dense ,epochs int) {
    diff := mat.NewDense(3,8, nil)
    for i:=0; i<2; i++{
        X_distorted.Product(X_distorted,W.T())
        X_distorted = Sgn(X_distorted)
        println("trained X")
        matPrint(X_distorted)
        diff.Sub(X,X_distorted)
        println("diff")
        matPrint(diff)
        println()
    }
}

func main(){
    X,X_distorted := createData()
    W := createWeights(X)
    println("Original X")
    matPrint(X)
    println("Distorted X")
    matPrint(X_distorted)
    println("Weights")
    matPrint(W)
    epochs := 2
    trainHopfieldBatch(X,X_distorted,W,epochs)
   }

