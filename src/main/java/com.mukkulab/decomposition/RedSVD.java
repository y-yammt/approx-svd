package com.mukkulab.decomposition;

import com.mukkulab.util.MatrixUtils;
import org.la4j.LinearAlgebra;
import org.la4j.Matrix;
import org.la4j.decomposition.MatrixDecompositor;

import java.util.Random;

public class RedSVD extends AbstractLowRankDecompositor implements MatrixDecompositor {
    public RedSVD(Matrix matrix) {
        super(matrix);
    }

    @Override
    public Matrix[] decompose(int rank) {
        int r = Math.min(rank, Math.min(matrix.rows(), matrix.columns()));
        if (r <= 0) {
            return new Matrix[]{
                    Matrix.zero(0, 0),
                    Matrix.zero(0, 0),
                    Matrix.zero(0, 0)
            };
        }

        Random random = new Random();

        // Gaussian Random Matrix for A^T
        Matrix o = MatrixUtils.randomGaussian(matrix.rows(), r, random);

        // Compute Sample Matrix of A^T
        Matrix y = matrix.transpose().multiply(o);

        // Orthonormalize Y
        y = MatrixUtils.processGramSchmidt(y);

        // Range(B) = Range(A^T)
        Matrix b = matrix.multiply(y);

        // Gaussian Random Matrix
        Matrix p = MatrixUtils.randomGaussian(b.columns(), r, random);

        // Compute Sample Matrix of B
        Matrix z = b.multiply(p);

        // Orthonormalize Z
        z = MatrixUtils.processGramSchmidt(z);

        // Range(C) = Range(B)
        Matrix c = z.transpose().multiply(b);

        // C = USV^T
        // A = Z * U * S * V^T * Y^T()
        Matrix svdOfC[] = LinearAlgebra.SVD.create(c).decompose();

        return new Matrix[] {
            z.multiply(svdOfC[0]),
            svdOfC[1],
            y.multiply(svdOfC[2])
        };
    }

    public boolean applicableTo(Matrix matrix) {
        return true;
    }
}
