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

func createData() (*mat.Dense, *mat.Dense, *mat.Dense) {
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
//     matPrint(X)
    W := mat.NewDense(8,8,nil)
    W.Product(X.T(),X)
    return X,X_dist,W
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

func main(){
    X,X_dist, W := createData()
    matPrint(X)
    println()
    matPrint(X_dist)
    println()
    matPrint(W)
    println()
    for i:=0; i<200; i++{
        X_dist.Product(X_dist,W.T())
        X_dist = Sgn(X_dist)
        matPrint(X_dist)
        println()
//         matPrint(W)
        println()
    }
}

