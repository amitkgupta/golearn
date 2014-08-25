// Package knn implements a K Nearest Neighbors object, capable of both classification
// and regression. It accepts data in the form of a slice of float64s, which are then reshaped
// into a X by Y matrix.
package knn

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/metrics/pairwise"
	"github.com/sjwhitworth/golearn/utilities"
	"runtime"
	"sync"
)

// A KNNClassifier consists of a data matrix, associated labels in the same order as the matrix, and a distance function.
// The accepted distance functions at this time are 'euclidean' and 'manhattan'.
type KNNClassifier struct {
	TrainingData      base.FixedDataGrid
	DistanceFunc      string
	NearestNeighbours int
}

// NewKnnClassifier returns a new classifier
func NewKnnClassifier(distfunc string, neighbours int) *KNNClassifier {
	KNN := KNNClassifier{}
	KNN.DistanceFunc = distfunc
	KNN.NearestNeighbours = neighbours
	return &KNN
}

// Fit stores the training data for later
func (KNN *KNNClassifier) Fit(trainingData base.FixedDataGrid) error {
	KNN.TrainingData = trainingData
	return nil
}

// Predict returns a classification for the vector, based on a vector input, using the KNN algorithm.
func (KNN *KNNClassifier) Predict(what base.FixedDataGrid) (base.FixedDataGrid, error) {

	// Check what distance function we are using
	var distanceFunc pairwise.PairwiseDistanceFunc
	switch KNN.DistanceFunc {
	case "euclidean":
		distanceFunc = pairwise.NewEuclidean()
	case "manhattan":
		distanceFunc = pairwise.NewManhattan()
	default:
		panic("unsupported distance function")
	}

	// Check compatability
	allAttrs := base.CheckCompatable(what, KNN.TrainingData)
	if allAttrs == nil {
		// Don't have the same Attributes
		return nil, nil
	}

	// Remove the Attributes which aren't numeric
	allNumericAttrs := make([]base.Attribute, 0)
	for _, a := range allAttrs {
		if fAttr, ok := a.(*base.FloatAttribute); ok {
			allNumericAttrs = append(allNumericAttrs, fAttr)
		}
	}
	numNumericalAttributes := len(allNumericAttrs)

	// Generate return vector
	ret := base.GeneratePredictionVector(what)

	// Resolve Attribute specifications for both
	whatAttrSpecs := base.ResolveAttributes(what, allNumericAttrs)
	trainAttrSpecs := base.ResolveAttributes(KNN.TrainingData, allNumericAttrs)

	// Reserve storage for row computations
	trainMat := [][]float64{}
	predMat := [][]float64{}

	// Iterate over all outer rows
	KNN.TrainingData.MapOverRows(trainAttrSpecs, func(trainRow [][]byte, srcRowNo int) (bool, error) {
		// Read the float values out
		trainMat = append(trainMat, make([]float64, numNumericalAttributes))
		for i, _ := range allNumericAttrs {
			trainMat[srcRowNo][i] = base.UnpackBytesToFloat(trainRow[i])
		}
		return true, nil
	})

	what.MapOverRows(whatAttrSpecs, func(predRow [][]byte, predRowNo int) (bool, error) {
		// Read the float values out
		predMat = append(predMat, make([]float64, numNumericalAttributes))
		for i, _ := range allNumericAttrs {
			predMat[predRowNo][i] = base.UnpackBytesToFloat(predRow[i])
		}

		return true, nil
	})

	runtime.GOMAXPROCS(runtime.NumCPU())
	wg := sync.WaitGroup{}
	wg.Add(len(predMat))
	for predRowNo, _ := range predMat {
		go func(p int) {
			defer wg.Done()

			predRow := predMat[p]
			distances := make(map[int]float64)

			// Find the closest match in the training data
			for srcRowNo, srcRow := range trainMat {
				v1 := mat64.NewDense(1, numNumericalAttributes, srcRow)
				v2 := mat64.NewDense(1, numNumericalAttributes, predRow)
				distances[srcRowNo] = distanceFunc.Distance(v1, v2)
			}

			sorted := utilities.SortIntMap(distances)
			values := sorted[:KNN.NearestNeighbours]

			maxmap := make(map[string]int)

			// Refresh maxMap
			for _, elem := range values {
				label := base.GetClass(KNN.TrainingData, elem)
				if _, ok := maxmap[label]; ok {
					maxmap[label]++
				} else {
					maxmap[label] = 1
				}
			}

			// Sort the maxMap
			var maxClass string
			maxVal := -1
			for a := range maxmap {
				if maxmap[a] > maxVal {
					maxVal = maxmap[a]
					maxClass = a
				}
			}

			base.SetClass(ret, p, maxClass)
		}(predRowNo)
	}
	wg.Wait()
	return ret, nil
}

// String returns a human-readable representation of this k-NN Classifier.
func (k *KNNClassifier) String() string {
	return fmt.Sprintf("KNNClassifier(NearestNeighbours: %d, DistanceFunc:%s\n)", k.NearestNeighbours, k.DistanceFunc)
}

// A KNNRegressor consists of a data matrix, associated result variables in the same order as the matrix, and a name.
type KNNRegressor struct {
	Data         *mat64.Dense
	Values       []float64
	DistanceFunc string
}

// NewKnnRegressor mints a new classifier.
func NewKnnRegressor(distfunc string) *KNNRegressor {
	KNN := KNNRegressor{}
	KNN.DistanceFunc = distfunc
	return &KNN
}

func (KNN *KNNRegressor) Fit(values []float64, numbers []float64, rows int, cols int) {
	if rows != len(values) {
		panic(mat64.ErrShape)
	}

	KNN.Data = mat64.NewDense(rows, cols, numbers)
	KNN.Values = values
}

func (KNN *KNNRegressor) Predict(vector *mat64.Dense, K int) float64 {
	// Get the number of rows
	rows, _ := KNN.Data.Dims()
	rownumbers := make(map[int]float64)
	labels := make([]float64, 0)

	// Check what distance function we are using
	var distanceFunc pairwise.PairwiseDistanceFunc
	switch KNN.DistanceFunc {
	case "euclidean":
		distanceFunc = pairwise.NewEuclidean()
	case "manhattan":
		distanceFunc = pairwise.NewManhattan()
	default:
		panic("unsupported distance function")
	}

	for i := 0; i < rows; i++ {
		row := KNN.Data.RowView(i)
		rowMat := utilities.FloatsToMatrix(row)
		distance := distanceFunc.Distance(rowMat, vector)
		rownumbers[i] = distance
	}

	sorted := utilities.SortIntMap(rownumbers)
	values := sorted[:K]

	var sum float64
	for _, elem := range values {
		value := KNN.Values[elem]
		labels = append(labels, value)
		sum += value
	}

	average := sum / float64(K)
	return average
}
