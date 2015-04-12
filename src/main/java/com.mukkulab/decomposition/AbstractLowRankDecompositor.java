package com.mukkulab.decomposition;

import org.la4j.Matrix;
import org.la4j.decomposition.MatrixDecompositor;

public abstract class AbstractLowRankDecompositor implements MatrixDecompositor {
    protected Matrix matrix;

    public AbstractLowRankDecompositor(Matrix matrix) {
        if (!applicableTo(matrix)) {
            fail("Given matrix can not be used with this decompositor.");
        }

        this.matrix = matrix;
    }

    abstract public Matrix[] decompose(int rank);

    public Matrix[] decompose() {
        return decompose(Math.min(matrix.rows(), matrix.columns()));
    }

    public Matrix self() {
        return matrix;
    }

    protected void fail(String message) {
        throw new IllegalArgumentException(message);
    }
}
