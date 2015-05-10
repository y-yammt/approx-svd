package com.mukkulab.decomposition;

import static org.junit.Assert.*;

import com.mukkulab.ApproxLinearAlgebra;
import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.sparse.CCSMatrix;

import java.util.Random;

public class RedSVDTest {
    private static final double EPS = 0.001;

    @Test
    public void redSvdTest() {
        //     +-         -+             +-           -+
        //     | 1 0 0 0 2 |             | 4 0       0 |
        // M = | 0 0 3 0 0 | ->  Sigma = | 0 3       0 |
        //     | 0 0 0 0 0 |    (rank3)  | 0 0 sqrt(5) |
        //     +-         -+             +-           -+
        ApproxLinearAlgebra.DecompositorFactory redSvdFactory = ApproxLinearAlgebra.DecompositorFactory.REDSVD;
        Matrix x = new Basic2DMatrix(new double[][]{
                new double[]{1, 0, 0, 0, 2},
                new double[]{0, 0, 3, 0, 0},
                new double[]{0, 0, 0, 0, 0},
                new double[]{0, 4, 0, 0, 0}
        });
        RedSVD redSvdComp = (RedSVD)redSvdFactory.create(x);
        Matrix[] svd = redSvdComp.decompose(3);

        // Check eigen values
        assertEquals(           4.0, svd[1].get(0, 0), EPS);
        assertEquals(           3.0, svd[1].get(1, 1), EPS);
        assertEquals(Math.sqrt(5.0), svd[1].get(2, 2), EPS);

        // Check orthogonality
        Matrix l = svd[0].transpose().multiply(svd[0]);
        for (int i = 0; i < l.rows(); ++i) {
            for (int j = 0; j < l.columns(); ++j) {
                assertEquals((i == j) ? 1.0 : 0.0, l.get(i, j), EPS);
            }
        }
        Matrix r = svd[2].transpose().multiply(svd[2]);
        for (int i = 0; i < r.rows(); ++i) {
            for (int j = 0; j < r.columns(); ++j) {
                assertEquals((i == j) ? 1.0 : 0.0, r.get(i, j), EPS);
            }
        }

        // Check composition matrix.
        Matrix c = svd[0].multiply(svd[1]).multiply(svd[2].transpose());
        for (int i = 0; i < x.rows(); ++i) {
            for (int j = 0; j < x.columns(); ++j) {
                assertEquals(x.get(i, j), c.get(i, j), EPS);
            }
        }
    }

    @Test
    public void redSvdRandomTest() {
        Random random = new Random();
        ApproxLinearAlgebra.DecompositorFactory redSvdFactory = ApproxLinearAlgebra.DecompositorFactory.REDSVD;

        int row = 10000;
        int col = 10000;
        Matrix sparseMatrix = new CCSMatrix(row, col);

        for (int i = 0; i < row; i += random.nextInt(1000)) {
            for (int j = 0; j < row; j += random.nextInt(1000)) {
                sparseMatrix.set(i, j, random.nextInt(10));
            }
        }
        RedSVD redSvdComp = (RedSVD)redSvdFactory.create(sparseMatrix);
        Matrix[] svd = redSvdComp.decompose(100);

        // Check composition matrix.
        Matrix c = svd[0].multiply(svd[1]).multiply(svd[2].transpose());
        for (int i = 0; i < sparseMatrix.rows(); ++i) {
            for (int j = 0; j < sparseMatrix.columns(); ++j) {
                assertEquals(sparseMatrix.get(i, j), c.get(i, j), EPS);
            }
        }
    }
}
