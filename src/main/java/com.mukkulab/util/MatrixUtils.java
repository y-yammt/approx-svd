package com.mukkulab.util;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.util.Random;

public class MatrixUtils {
    private static final double SVD_EPS = 0.0001f;

    /**
     * Creates a gaussian distributed random {@link Basic2DMatrix} of the given shape:
     * {@code rows} x {@code columns}.
     */
    public static Basic2DMatrix randomGaussian(int rows, int columns, Random random) {
        double array[][] = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                array[i][j] = random.nextGaussian();
            }
        }
        return new Basic2DMatrix(array);
    }

    /**
     *
     * @param matrix
     * @return
     */
    public static Matrix processGramSchmidt(Matrix matrix) {
        Matrix y = matrix.blankOfShape(matrix.rows(), matrix.columns());
        for (int i = 0; i < matrix.columns(); ++i) {
            Vector v = matrix.getColumn(i);
            for (int j = 0; j < i; ++j) {
                double r = v.innerProduct(y.getColumn(j));
                v = v.subtract(y.getColumn(j).multiply(r));
            }
            double norm = v.norm();
            if (norm < SVD_EPS) {
                break;
            }
            y.setColumn(i, v.multiply(1.f / norm));
        }
        return y;
    }
}
