package com.mukkulab.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.util.Random;

public class MatrixUtilTest {
    private static final double EPS = 0.001;

    // TODO: variance check
    @Test
    public void randomGaussianRandomTest() {
        Random random = new Random();
        Matrix x = MatrixUtils.randomGaussian(1000, 1000, random);
        int n = 0;
        double avg = 0.0;
        for (double e : x) {
            ++n;
            avg = avg + (1.0 / n) * (e - avg);
        }
        assertEquals(0.0, avg, EPS);
    }

    @Test
    public void processGramSchmidtTest() {
        // Case 1
        //   +-     -+    +-                        -+
        //   | 3 2 1 | -> | 3/sqrt(10) -1/sqrt(10) 0 |
        //   | 1 2 1 |    | 1/sqrt(10)  3/sqrt(10) 0 |
        //   +-     -+    +-                        -+
        Matrix x = new Basic2DMatrix(new double[][]{
                new double[]{3, 2, 1},
                new double[]{1, 3, 1}
        });
        Matrix normX = MatrixUtils.processGramSchmidt(x);
        assertEquals(  3.0 / Math.sqrt(10.0), normX.get(0, 0), EPS);
        assertEquals(  1.0 / Math.sqrt(10.0), normX.get(1, 0), EPS);
        assertEquals(- 1.0 / Math.sqrt(10.0), normX.get(0, 1), EPS);
        assertEquals(  3.0 / Math.sqrt(10.0), normX.get(1, 1), EPS);
        assertEquals(                    0.0, normX.get(0, 2), EPS);
        assertEquals(                    0.0, normX.get(1, 2), EPS);

        // Case 2
        //   +-     -+    +-                                      -+
        //   | 1 1 0 | -> | 1/sqrt(2)  sqrt(2)/2sqrt(3) -sqrt(3)/3 |
        //   | 1 0 1 |    | 1/sqrt(2) -sqrt(2)/2sqrt(3)  sqrt(3)/3 |
        //   | 0 1 1 |    |         0  sqrt(2)/ sqrt(3)  sqrt(3)/3 |
        //   +-     -+    +-                                      -+
        x = new Basic2DMatrix(new double[][]{
                new double[]{1, 1, 0},
                new double[]{1, 0, 1},
                new double[]{0, 1, 1}
        });
        normX = MatrixUtils.processGramSchmidt(x);
        assertEquals(                     1.0 / Math.sqrt(2.0), normX.get(0, 0), EPS);
        assertEquals(                     1.0 / Math.sqrt(2.0), normX.get(1, 0), EPS);
        assertEquals(                                      0.0, normX.get(2, 0), EPS);
        assertEquals(  Math.sqrt(2.0) / (2.0 * Math.sqrt(3.0)), normX.get(0, 1), EPS);
        assertEquals(- Math.sqrt(2.0) / (2.0 * Math.sqrt(3.0)), normX.get(1, 1), EPS);
        assertEquals(          Math.sqrt(2.0) / Math.sqrt(3.0), normX.get(2, 1), EPS);
        assertEquals(                   - Math.sqrt(3.0) / 3.0, normX.get(0, 2), EPS);
        assertEquals(                     Math.sqrt(3.0) / 3.0, normX.get(1, 2), EPS);
        assertEquals(                     Math.sqrt(3.0) / 3.0, normX.get(2, 2), EPS);
    }
}
