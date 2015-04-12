package com.mukkulab;

import com.mukkulab.decomposition.RedSVD;
import org.la4j.LinearAlgebra;
import org.la4j.Matrix;
import org.la4j.decomposition.MatrixDecompositor;

/**
 * Tiny class for common things.
 */
public final class ApproxLinearAlgebra {
    /**
     * The library version.
     */
    public static final String VERSION = "0.0.1-SNAPSHOT";

    /**
     * The library name.
     */
    public static final String NAME = "approxsvd-java";

    /**
     * The library release date.
     */
    public static final String DATE = "March 2015";

    /**
     * The library full name.
     */
    public static final String FULL_NAME = NAME + "-" + VERSION + " (" + DATE + ")";

    /**
     * The machine epsilon, which is calculated at runtime.
     */
    public static final double EPS = LinearAlgebra.EPS;

    /**
     * Exponent of machine epsilon
     */
    public static final int ROUND_FACTOR = LinearAlgebra.ROUND_FACTOR;

    public static enum DecompositorFactory {
        REDSVD {
            @Override
            public MatrixDecompositor create(Matrix matrix) {
                return new RedSVD(matrix);
            }
        };

        public abstract MatrixDecompositor create(Matrix matrix);
    }

    /**
     * Reference to SVD decompositor factory.
     */
    public static final DecompositorFactory REDSVD = DecompositorFactory.REDSVD;
}
