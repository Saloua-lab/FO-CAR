package LDOS1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import util.Pair;

/**
 * Algorithm 1 — SHAPLEY-BASED CONTEXT FEATURE WEIGHTING (FO-CAR)
 *
 * Computes Shapley values for each contextual dimension using Monte Carlo
 * permutation sampling (Castro et al., 2009).
 *
 * For LDOS-CoMoDa, the 5 contextual dimensions are:
 *   [0] time, [1] weather, [2] social, [3] mood, [4] location
 *
 * Output: shapleyWeights[i] = importance of dimension i in [0,1]
 */
public class ShapleyWeighting {

    // Number of Monte Carlo iterations (100 is sufficient per paper)
    public static int N_ITER = 100;

    // Number of contextual dimensions (5 for LDOS)
    public static int N_DIMS = 5;

    // Random seed for reproducibility
    public static long SHAPLEY_SEED = 42L;

    /**
     * Main entry point.
     * Computes Shapley weights for all contextual dimensions on trainData.
     *
     * @param trainData  training data (LdosData without test pairs)
     * @param targetUser target user id
     * @param movieId    target item id
     * @param baselineMAE MAE of the context-unaware baseline (UserKNN)
     * @return ArrayList<Float> of size N_DIMS with Shapley weights
     */
    public static ArrayList<Float> computeShapleyWeights(LdosData trainData,
                                                          int targetUser,
                                                          int movieId,
                                                          float baselineMAE) {

        // Initialize Shapley accumulators
        double[] shapleySum = new double[N_DIMS];
        int[]    shapleyCount = new int[N_DIMS];

        Random rnd = new Random(SHAPLEY_SEED);

        // Features indices: [0..N_DIMS-1]
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < N_DIMS; i++) allFeatures.add(i);

        for (int iter = 0; iter < N_ITER; iter++) {

            // Sample a random permutation of features (without replacement)
            List<Integer> perm = new ArrayList<>(allFeatures);
            Collections.shuffle(perm, rnd);

            // For each feature in the permutation, compute marginal contribution
            List<Integer> coalition = new ArrayList<>();

            for (int pos = 0; pos < perm.size(); pos++) {
                int feature = perm.get(pos);

                // v(S) = utility of coalition WITHOUT feature
                double vWithout = computeCoalitionUtility(
                        trainData, targetUser, movieId, coalition, baselineMAE);

                // v(S ∪ {feature}) = utility WITH feature
                List<Integer> coalitionWith = new ArrayList<>(coalition);
                coalitionWith.add(feature);
                double vWith = computeCoalitionUtility(
                        trainData, targetUser, movieId, coalitionWith, baselineMAE);

                // Marginal contribution
                double marginal = vWith - vWithout;
                shapleySum[feature]   += marginal;
                shapleyCount[feature] += 1;

                // Add feature to coalition for next step
                coalition.add(feature);
            }
        }

        // Average marginal contributions → Shapley values
        double[] shapley = new double[N_DIMS];
        for (int i = 0; i < N_DIMS; i++) {
            shapley[i] = (shapleyCount[i] > 0) ? shapleySum[i] / shapleyCount[i] : 0.0;
        }

        // Normalize to [0,1] — negative values clamped to 0
        double maxVal = 0.0;
        for (int i = 0; i < N_DIMS; i++) {
            if (shapley[i] > maxVal) maxVal = shapley[i];
        }

        ArrayList<Float> weights = new ArrayList<>();
        for (int i = 0; i < N_DIMS; i++) {
            double w = (maxVal > 0) ? Math.max(0.0, shapley[i] / maxVal) : (1.0 / N_DIMS);
            weights.add((float) w);
        }

        return weights;
    }
    private static int getAvgRatingFromSituations(ArrayList<ArrayList<Integer>> situations) {
        if (situations == null || situations.isEmpty()) return -1;

        double sum = 0;
        int n = 0;
        for (ArrayList<Integer> one : situations) {
            if (one != null && !one.isEmpty()) {
                int r = one.get(0);
                if (r != -1) { sum += r; n++; }
            }
        }
        if (n == 0) return -1;
        return (int) Math.round(sum / n);
    }
    static int getRating(int userId, int movieId, LdosData data) {
        Pair p = new Pair(userId, movieId);
        ArrayList<ArrayList<Integer>> situations = data.UserMovieRatings.get(p);
        if (situations == null || situations.isEmpty()) return -1;
        return getAvgRatingFromSituations(situations);
    }

    /**
     * Computes v(S) = utility of using only features in 'coalition' for prediction.
     *
     * v(S) = |r_real - r_baseline| - |r_real - r_predicted_with_S|
     *      = reduction in prediction error due to coalition S
     *
     * If coalition is empty → returns 0 (baseline, no improvement).
     */
    private static double computeCoalitionUtility(LdosData trainData,
                                                   int targetUser,
                                                   int movieId,
                                                   List<Integer> coalition,
                                                   float baselineMAE) {

        if (coalition == null || coalition.isEmpty()) return 0.0;

        // Build a restricted weight vector: 1.0 for features in coalition, 0 otherwise
        ArrayList<Float> vRestricted = new ArrayList<>();
        for (int i = 0; i < N_DIMS; i++) {
            vRestricted.add(coalition.contains(i) ? 1.0f : 0.0f);
        }

        // Get real rating for targetUser/movieId
        int realRating = getRating(targetUser, movieId, trainData);
        if (realRating == -1) return 0.0;

        // Predict using restricted features
        float predicted = predictWithRestrictedFeatures(
                trainData, targetUser, movieId, vRestricted);
        if (predicted == -1) return 0.0;

        // v(S) = reduction in absolute error
        double errorBaseline   = Math.abs(realRating - baselineMAE);
        double errorCoalition  = Math.abs(realRating - predicted);
        return errorBaseline - errorCoalition;
    }

    /**
     * Predicts rating using only the contextual features in vRestricted (non-zero = active).
     * Uses the same kNN mechanism as Neighbor2 but with restricted weights.
     */
    private static float predictWithRestrictedFeatures(LdosData trainData,
                                                        int targetUser,
                                                        int movieId,
                                                        ArrayList<Float> vRestricted) {

        try {
            // Get context-based neighbors with restricted features
            Map<Integer, util.ContextNeighSim> neighbors =
                    Neighbor2.NeighborSelection(targetUser, movieId, trainData, vRestricted);

            if (neighbors == null || neighbors.isEmpty()) {
                // fallback: global mean
                return computeGlobalMean(trainData);
            }

            // Weighted average of neighbor ratings
            float sumSim = 0f;
            float sumRat = 0f;
            for (Map.Entry<Integer, util.ContextNeighSim> e : neighbors.entrySet()) {
                float sim    = e.getValue().Sim;
                float rating = e.getValue().rating;
                sumSim += sim;
                sumRat += sim * rating;
            }
            if (sumSim == 0) return computeGlobalMean(trainData);
            return sumRat / sumSim;

        } catch (Exception ex) {
            return computeGlobalMean(trainData);
        }
    }

    /**
     * Global mean rating over all training data (context-unaware baseline).
     */
    public static float computeGlobalMean(LdosData trainData) {
        double sum = 0; int n = 0;
        for (ArrayList<ArrayList<Integer>> sits : trainData.UserMovieRatings.values()) {
            if (sits == null) continue;
            for (ArrayList<Integer> s : sits) {
                if (s != null && !s.isEmpty() && s.get(0) != -1) {
                    sum += s.get(0); n++;
                }
            }
        }
        return (n > 0) ? (float)(sum / n) : 3.0f;
    }

    /**
     * Compute context-unaware baseline MAE (UserKNN without context).
     * Used as v(∅) reference.
     */
    public static float computeBaselineMAE(LdosData trainData, List<Pair> testPairs) {
        // uniform weights (all features equally weighted → additive baseline)
        ArrayList<Float> uniformV = new ArrayList<>();
        for (int i = 0; i < N_DIMS; i++) uniformV.add(1.0f / N_DIMS);

        float sumErr = 0; int n = 0;
        for (Pair p : testPairs) {
            ArrayList<ArrayList<Integer>> sitsTest = trainData.UserMovieRatings.get(p);
            if (sitsTest == null || sitsTest.isEmpty()) continue;

            int real = 0; int cnt = 0;
            for (ArrayList<Integer> s : sitsTest) {
                if (s != null && !s.isEmpty() && s.get(0) != -1) { real += s.get(0); cnt++; }
            }
            if (cnt == 0) continue;
            real = Math.round((float) real / cnt);

            float pred = predictWithRestrictedFeatures(
                    trainData, p.UserId, p.Id, new ArrayList<Float>());
            if (pred == -1) continue;

            sumErr += Math.abs(real - pred); n++;
        }
        return (n > 0) ? sumErr / n : 1.0f;
    }
}
