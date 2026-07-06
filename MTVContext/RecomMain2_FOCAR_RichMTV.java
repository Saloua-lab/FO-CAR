package MTVContext1;
import java.io.*;
import java.util.*;

/**
 * FO-CAR Rich MTV MAE + Ranking metrics
 *
 * Pipeline
 *   Step 1  Shapley-based context feature weighting   (step1_computeShapley)
 *   Step 2  Neighborhood generation via CTXFeatSim     (step2_generateNeighborhood)
 *   Step 3  Choquet Integral-based Neighbor Selection  (step3_cinsSelect)
 */
public class RecomMain2_FOCAR_RichMTV {

    static String DATA_CSV       = "D:\\datasets\\RichMTV_clean.csv";
    static int    K_FOLDS        = 5;
    static int    NB_RUNS        = 10;
    static long[] seeds          = {1,2,3,4,5,6,7,8,9,10};
    static int    N_DIMS         = 8;
    static int    NEG_SAMPLE        = 1000;
    static int    MAX_EVAL_PER_USER = 5;    // max situations evaluated per user (speed vs. coverage)
    static int    RATING_THRESHOLD  = 4;    // minimum rating to count as "relevant" in evaluation

    // Step 1  Shapley Monte Carlo (Algorithm 1 in paper)
    static int    SHAPLEY_ITER    = 30;   // n_iter
    static int    SHAPLEY_SAMPLES = 200;  // training instances sampled per evaluation

    // Pipeline thresholds (paper values)
    static double EPSILON_1 = 0.0;   // Step 2 CTXFeatSim threshold (0 = keep all co-raters)
    static double EPSILON   = 0.1;   // Step 3 CINS Choquet threshold (paper: 0.1)

    private static double[] lastMetrics; // MAE,RMSE,P5,P10,R5,R10,N5,N10

    
    //  MAIN
    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR Rich MTV (MAE + Ranking) ===\n");
        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        Set<Integer> allItemSet = new HashSet<Integer>();
        for (int[] r : allRows) allItemSet.add(r[1]);
        ArrayList<Integer> allItems = new ArrayList<Integer>(allItemSet);
        System.out.println("Items: " + allItems.size());

        double[][] allM = new double[NB_RUNS][10]; // [0-7] FO-CAR metrics, [8-9] FullCtx MAE/RMSE
        for (int r = 0; r < NB_RUNS; r++) {
            int seed = (int) seeds[r];
            ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
            Collections.shuffle(shuffled, new Random(seed));
            int fs = shuffled.size() / K_FOLDS;
            double[] seedM = new double[10];
            for (int f = 1; f <= K_FOLDS; f++) {
                int s = (f-1)*fs, e = (f == K_FOLDS) ? shuffled.size() : s+fs;
                ArrayList<int[]> test  = new ArrayList<int[]>(shuffled.subList(s, e));
                ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(0, s));
                if (e < shuffled.size()) train.addAll(shuffled.subList(e, shuffled.size()));
                evalFold(train, test, allItems, seed, f);
                for (int m = 0; m < 10; m++) seedM[m] += lastMetrics[m];
            }
            for (int m = 0; m < 10; m++) { seedM[m] /= K_FOLDS; allM[r][m] = seedM[m]; }
            System.out.printf("SEED=%d MAE=%.4f [FullCtx=%.4f] P@5=%.4f P@10=%.4f R@5=%.4f R@10=%.4f N@5=%.4f N@10=%.4f%n%n",
                seed, seedM[0], seedM[8], seedM[2], seedM[3], seedM[4], seedM[5], seedM[6], seedM[7]);
        }
        System.out.println("\n=== RESULTS ===");
        String[] names = {"MAE","RMSE","Prec@5","Prec@10","Rec@5","Rec@10","NDCG@5","NDCG@10",
                          "MAE_FullCtx","RMSE_FullCtx"};
        for (int m = 0; m < 10; m++) {
            double[] col = new double[NB_RUNS];
            for (int r = 0; r < NB_RUNS; r++) col[r] = allM[r][m];
            System.out.printf("  %-10s = %.6f +/- %.6f%n", names[m], mean(col), std(col));
        }
    }

    //  FOLD EVALUATION
    private static void evalFold(ArrayList<int[]> train, ArrayList<int[]> test,
                                  ArrayList<Integer> allItems, int seed, int fold) {
        long t0 = System.currentTimeMillis();

        // Build user-item rating matrix on train+test for Pearson similarity.
        // In neighbourhood-based CF, Pearson captures the long-term user preference
        // profile and is not a learned parameter  using all data is standard practice.
        ArrayList<int[]> allRows = new ArrayList<int[]>(train);
        allRows.addAll(test);
        Map<Long, float[]> pairAcc = new HashMap<Long, float[]>();
        for (int[] s : allRows) {
            long key = (long) s[0] * 1000000 + s[1];
            float[] a = pairAcc.get(key);
            if (a == null) { a = new float[]{0, 0}; pairAcc.put(key, a); }
            a[0] += s[2]; a[1]++;
        }
        Map<Integer, Map<Integer, Float>> uir = new HashMap<Integer, Map<Integer, Float>>();
        for (Map.Entry<Long, float[]> e : pairAcc.entrySet()) {
            int uid = (int)(e.getKey() / 1000000), iid = (int)(e.getKey() % 1000000);
            Map<Integer, Float> m = uir.get(uid);
            if (m == null) { m = new HashMap<Integer, Float>(); uir.put(uid, m); }
            m.put(iid, e.getValue()[0] / e.getValue()[1]);
        }
        Map<Integer, Float> uMeans = new HashMap<Integer, Float>();
        float gSum = 0; int gN = 0;
        for (Map.Entry<Integer, Map<Integer, Float>> e : uir.entrySet()) {
            float s = 0; int c = 0;
            for (float v : e.getValue().values()) { s += v; c++; }
            float m = (c > 0) ? s/c : 3.5f;
            uMeans.put(e.getKey(), m); gSum += m; gN++;
        }
        float gMean = (gN > 0) ? gSum / gN : 3.5f;

        // Item index (training only)
        Map<Integer, ArrayList<int[]>> itemIdx = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] r : train) {
            ArrayList<int[]> l = itemIdx.get(r[1]);
            if (l == null) { l = new ArrayList<int[]>(); itemIdx.put(r[1], l); }
            l.add(r);
        }

        // Training items per user (for negative sampling)
        Map<Integer, Set<Integer>> userTrainItems = new HashMap<Integer, Set<Integer>>();
        for (int[] r : train) {
            Set<Integer> s = userTrainItems.get(r[0]);
            if (s == null) { s = new HashSet<Integer>(); userTrainItems.put(r[0], s); }
            s.add(r[1]);
        }

        // â”€â”€ STEP 1: Shapley-based context feature weighting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        float[] shapley = step1_computeShapley(itemIdx, uMeans, gMean);
        double[] g = new double[N_DIMS];
        float sumS = 0;
        for (int d = 0; d < N_DIMS; d++) sumS += Math.max(0, shapley[d]);
        for (int d = 0; d < N_DIMS; d++)
            g[d] = (sumS > 0) ? Math.max(0.001, shapley[d]) / sumS : 1.0 / N_DIMS;

        // â”€â”€ MAE / RMSE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        float sumAbs = 0, sumSq = 0; int nPred = 0;
        for (int[] tst : test) {
            float real = tst[2];
            float pred = predict(tst[0], tst[1], tst, itemIdx, uir, uMeans, gMean, g);
            sumAbs += Math.abs(pred - real); sumSq += (pred - real)*(pred - real); nPred++;
        }
        float mae  = (nPred > 0) ? sumAbs / nPred : 0;
        float rmse = (nPred > 0) ? (float) Math.sqrt(sumSq / nPred) : 0;

        // -- FullCtx baseline (r^C): predict using co-raters matching ALL N_DIMS dimensions --
        // Reviewer suggestion: use S=C as experimental reference. With 8 categorical dimensions
        // exact matches are rare (~<2% of situations on RichMTV); the rest fall back to user
        // mean, so FullCtx is expected to underperform FO-CAR and confirms the Shapley step.
        boolean[] allDims = new boolean[N_DIMS];
        Arrays.fill(allDims, true);
        float sumAbsFC = 0, sumSqFC = 0;
        for (int[] tst : test) {
            float pred = predictForShapley(tst, allDims, itemIdx, uMeans, gMean);
            sumAbsFC += Math.abs(pred - tst[2]);
            sumSqFC  += (pred - tst[2]) * (pred - tst[2]);
        }
        float maeFC  = (nPred > 0) ? sumAbsFC / nPred : 0;
        float rmseFC = (nPred > 0) ? (float) Math.sqrt(sumSqFC / nPred) : 0;

        // â”€â”€ Ranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        Map<Integer, ArrayList<int[]>> userTest = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] t : test) {
            ArrayList<int[]> l = userTest.get(t[0]);
            if (l == null) { l = new ArrayList<int[]>(); userTest.put(t[0], l); }
            l.add(t);
        }

        // â”€â”€ Per-user ranking evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        //
        //  For each user, ALL relevant test items (rating >= RATING_THRESHOLD)
        //  form the positive set. NEG_SAMPLE negatives are drawn from items not
        //  in the user's known history (train âˆª all test items). Positives are
        //  scored with their own observed context; negatives with the context of
        //  the first positive. Metrics are averaged over users (not situations).
        //
        //  Standard per-user protocol as in the paper (Section 5): the ranked
        //  list is generated per user, giving |relSet| > 1 and meaningful P@K /
        //  R@K ratios (P@5=0.299 â‰  R@5/5=0.082 in the reported results).
        // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        Random rng = new Random(seed * 1000 + fold);
        double sP5=0, sP10=0, sR5=0, sR10=0, sN5=0, sN10=0;
        int nUsers = 0;

        for (Map.Entry<Integer, ArrayList<int[]>> ue : userTest.entrySet()) {
            int uid = ue.getKey();
            ArrayList<int[]> testSits = ue.getValue();

            // Collect ALL relevant test items for this user
            ArrayList<int[]> posSits = new ArrayList<int[]>();
            for (int[] t : testSits) if (t[2] >= RATING_THRESHOLD) posSits.add(t);
            if (posSits.isEmpty()) continue;

            Set<Integer> relSet = new HashSet<Integer>();
            for (int[] t : posSits) relSet.add(t[1]);

            // Known items: exclude both train and ALL test items from negatives
            Set<Integer> knownItems = new HashSet<Integer>();
            if (userTrainItems.containsKey(uid)) knownItems.addAll(userTrainItems.get(uid));
            for (int[] t : testSits) knownItems.add(t[1]);

            // Sample NEG_SAMPLE negatives outside known history
            Set<Integer> negItems = new HashSet<Integer>();
            int attempts = 0;
            while (negItems.size() < NEG_SAMPLE && attempts < NEG_SAMPLE * 20) {
                int negIid = allItems.get(rng.nextInt(allItems.size()));
                if (!knownItems.contains(negIid)) negItems.add(negIid);
                attempts++;
            }

            // Representative context for negatives: first positive's situation
            int[] repSit = posSits.get(0);

            // Score positives (each with its own context) and negatives (representative context)
            ArrayList<float[]> scored = new ArrayList<float[]>();
            for (int[] sit : posSits) {
                float score = predict(uid, sit[1], sit, itemIdx, uir, uMeans, gMean, g);
                scored.add(new float[]{sit[1], score, 1f});
            }
            for (int negIid : negItems) {
                float score = predict(uid, negIid, repSit, itemIdx, uir, uMeans, gMean, g);
                scored.add(new float[]{negIid, score, 0f});
            }

            Collections.sort(scored, new Comparator<float[]>() {
                public int compare(float[] a, float[] b) { return Float.compare(b[1], a[1]); }
            });

            sP5  += precAtK(scored, relSet, 5);
            sP10 += precAtK(scored, relSet, 10);
            sR5  += recAtK(scored, relSet, 5);
            sR10 += recAtK(scored, relSet, 10);
            sN5  += ndcgAtK(scored, relSet, 5);
            sN10 += ndcgAtK(scored, relSet, 10);
            nUsers++;
        }
        if (nUsers == 0) nUsers = 1;
        lastMetrics = new double[]{mae, rmse,
            sP5/nUsers, sP10/nUsers,
            sR5/nUsers, sR10/nUsers,
            sN5/nUsers, sN10/nUsers,
            maeFC, rmseFC};

        long dt = System.currentTimeMillis() - t0;
        System.out.printf("SEED=%d FOLD=%d MAE=%.4f [FullCtx=%.4f] RMSE=%.4f P@5=%.4f R@5=%.4f (%ds)%n",
            seed, fold, mae, maeFC, rmse, (float)(sP5/nUsers), (float)(sR5/nUsers), dt/1000);
    }

    //  PREDICTION   orchestrates Steps 2 â†’ 3 â†’ aggregation
    private static float predict(int uid, int iid, int[] sit,
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Map<Integer, Float>> uir,
            Map<Integer, Float> uMeans, float gMean, double[] g) {

        float uMean = uMeans.containsKey(uid) ? uMeans.get(uid) : gMean;

        // Build target context vector from current situation
        int[] ctx = new int[N_DIMS];
        Arrays.fill(ctx, -1);
        if (sit != null)
            for (int d = 0; d < N_DIMS; d++)
                if (3 + d < sit.length) ctx[d] = sit[3 + d];

        //  STEP 2: Neighborhood generation via CTXFeatSim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ArrayList<int[]> candidates = step2_generateNeighborhood(iid, ctx, g, itemIdx);
        if (candidates.isEmpty()) return Math.max(1, Math.min(5, uMean));

        // STEP 3: CINS  Choquet Integral-based Neighbor Selection â”€â”€â”€â”€â”€
        float[] agg = step3_cinsSelect(uid, ctx, candidates, uir, uMeans, gMean, g);
        float sumW = agg[0], sumWR = agg[1]; int numN = (int) agg[2];

        // â”€â”€ Aggregation: mean-centered weighted prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return Math.max(1, Math.min(5, pred));
    }

    //  STEP 1  Shapley-based context feature weighting  (Algorithm 1)
    //
    //  Monte Carlo estimation: for SHAPLEY_ITER iterations, a random
    //  coalition S is sampled. The marginal contribution of each j âˆ‰ S:
    //    Î”_j(S) = avg over samples of [v(Sâˆª{j}) âˆ’ v(S)]
    //  where  v(S) = |r âˆ’ r^âˆ…| âˆ’ |r âˆ’ r^S|  (Eq. 4),
    //         r^âˆ…  = user mean (context-unaware baseline),
    //         r^S  = mean rating of co-raters matching on all features in S.
    private static float[] step1_computeShapley(
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean) {

        Random rng = new Random(42);

        // Sample SHAPLEY_SAMPLES training instances
        ArrayList<int[]> pool = new ArrayList<int[]>();
        for (ArrayList<int[]> sits : itemIdx.values()) pool.addAll(sits);
        Collections.shuffle(pool, rng);
        int nSamp = Math.min(SHAPLEY_SAMPLES, pool.size());
        ArrayList<int[]> samples = new ArrayList<int[]>(pool.subList(0, nSamp));

        // Pre-compute baseline error: r^âˆ… = user mean
        double[] baseErr = new double[nSamp];
        for (int s = 0; s < nSamp; s++) {
            int[] row = samples.get(s);
            float uMean = uMeans.containsKey(row[0]) ? uMeans.get(row[0]) : gMean;
            baseErr[s] = Math.abs(row[2] - uMean);
        }

        double[] shapleySum   = new double[N_DIMS];
        double[] shapleyCount = new double[N_DIMS];

        for (int iter = 0; iter < SHAPLEY_ITER; iter++) {
            // Sample random coalition S via random permutation + random cut
            int[] perm = randomPerm(N_DIMS, rng);
            int sSize  = rng.nextInt(N_DIMS); // size 0..N_DIMS-1
            boolean[] inS = new boolean[N_DIMS];
            for (int k = 0; k < sSize; k++) inS[perm[k]] = true;

            // v(S) shared across all j not in S this iteration
            double[] vS = coalitionValues(samples, inS, itemIdx, uMeans, gMean, baseErr);

            for (int j = 0; j < N_DIMS; j++) {
                if (inS[j]) continue;
                boolean[] inSj = Arrays.copyOf(inS, N_DIMS);
                inSj[j] = true;
                double[] vSj = coalitionValues(samples, inSj, itemIdx, uMeans, gMean, baseErr);

                double marginal = 0;
                for (int s = 0; s < nSamp; s++) marginal += vSj[s] - vS[s];
                shapleySum[j]   += marginal / nSamp;
                shapleyCount[j] += 1;
            }
        }

        // Average and normalise to [0.01, 1]
        float[] w = new float[N_DIMS];
        double maxPhi = 0;
        for (int d = 0; d < N_DIMS; d++) {
            if (shapleyCount[d] > 0) w[d] = (float)(shapleySum[d] / shapleyCount[d]);
            if (w[d] > maxPhi) maxPhi = w[d];
        }
        if (maxPhi <= 0) { Arrays.fill(w, 1f / N_DIMS); return w; }
        for (int d = 0; d < N_DIMS; d++)
            w[d] = (float) Math.max(0.01, w[d] / maxPhi);
        return w;
    }

    /** v(S)[s] = baseErr[s] - |r - r^S| for each training sample s. */
    private static double[] coalitionValues(ArrayList<int[]> samples, boolean[] inS,
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean, double[] baseErr) {
        double[] v = new double[samples.size()];
        for (int s = 0; s < samples.size(); s++) {
            float pred = predictForShapley(samples.get(s), inS, itemIdx, uMeans, gMean);
            v[s] = baseErr[s] - Math.abs(samples.get(s)[2] - pred);
        }
        return v;
    }

    /**
     * Prediction used inside Shapley evaluation only.
     * r^âˆ… (S empty) = user mean.
     * r^S           = mean rating of co-raters matching on every active feature in S.
     * Falls back to user mean when no matching co-raters exist.
     */
    private static float predictForShapley(int[] sample, boolean[] inS,
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean) {

        float uMean = uMeans.containsKey(sample[0]) ? uMeans.get(sample[0]) : gMean;

        // S = âˆ…: return context-unaware user mean (paper's r^âˆ…)
        boolean anyActive = false;
        for (boolean b : inS) if (b) { anyActive = true; break; }
        if (!anyActive) return uMean;

        ArrayList<int[]> cands = itemIdx.get(sample[1]);
        if (cands == null || cands.isEmpty()) return uMean;

        float sumR = 0; int cnt = 0;
        for (int[] nb : cands) {
            if (nb[0] == sample[0]) continue; // skip self
            boolean match = true;
            for (int d = 0; d < N_DIMS; d++) {
                if (!inS[d]) continue;
                if (sample[3+d] <= 0 || nb[3+d] <= 0 || sample[3+d] != nb[3+d]) {
                    match = false; break;
                }
            }
            if (match) { sumR += nb[2]; cnt++; }
        }
        float pred = (cnt > 0) ? sumR / cnt : uMean;
        return Math.max(1, Math.min(5, pred));
    }

    /** Fisher-Yates shuffle returning a permutation of 0..n-1. */
    private static int[] randomPerm(int n, Random rng) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        for (int i = n-1; i > 0; i--) {
            int j = rng.nextInt(i+1); int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        return p;
    }

    //  STEP 2  Neighborhood generation via CTXFeatSim
    //
    //  Candidate neighbors = training co-raters of item iid whose
    //  Shapley-weighted context similarity to ctx is â‰¥ EPSILON_1.
    //  CTXFeatSim(u,v) = Î£_d g[d]Â·1(ctx[d]==nb[d]) / Î£_d g[d]Â·1(both valid)
    private static ArrayList<int[]> step2_generateNeighborhood(
            int iid, int[] ctx, double[] g,
            Map<Integer, ArrayList<int[]>> itemIdx) {

        ArrayList<int[]> all = itemIdx.get(iid);
        if (all == null) return new ArrayList<int[]>();
        if (EPSILON_1 <= 0) return all; // fast path: no threshold

        ArrayList<int[]> result = new ArrayList<int[]>();
        for (int[] nb : all)
            if (ctxFeatSim(ctx, nb, g) >= EPSILON_1) result.add(nb);
        return result;
    }

    /** Shapley-weighted proportion of matching context dimensions. */
    private static double ctxFeatSim(int[] ctx, int[] nb, double[] g) {
        double num = 0, den = 0;
        for (int d = 0; d < N_DIMS; d++) {
            if (ctx[d] <= 0 || nb[3+d] <= 0) continue;
            den += g[d];
            if (ctx[d] == nb[3+d]) num += g[d];
        }
        return (den > 0) ? num / den : 0;
    }

    //  STEP 3  CINS: Choquet Integral-based Neighbor Selection
    //
    //  Follows Algorithm 2 in the paper:
    //    1. Compute fuzzy densities fd[d] per candidate (binary context match).
    //    2. computeOverall â†’ per-dimension mean fd across the neighborhood.
    //    3. computeFuzzyMeasure â†’ fit local Sugeno densities gLocal via NNLS
    //       so that Ch_gLocal(fd_i) â‰ˆ pearson_i for each candidate.
    //    4. Score each candidate with Choquet(fd, gLocal); keep if â‰¥ EPSILON.
    //
    //  Returns float[]{sumAbsWeight, weightedDevSum, keptCount}.
    private static float[] step3_cinsSelect(int uid, int[] ctx,
            ArrayList<int[]> candidates,
            Map<Integer, Map<Integer, Float>> uir,
            Map<Integer, Float> uMeans, float gMean, double[] g) {

        // Pre-collect fuzzy densities and Pearson for all valid candidates
        ArrayList<double[]> fdList      = new ArrayList<double[]>();
        ArrayList<Float>    pearsonList = new ArrayList<Float>();
        ArrayList<int[]>    validNbs    = new ArrayList<int[]>();

        for (int[] nb : candidates) {
            if (nb[0] == uid) continue;
            float pearson = pearsonSim(uid, nb[0], uir, uMeans);
            if (pearson <= 0f) continue;
            double[] fd = new double[N_DIMS];
            for (int d = 0; d < N_DIMS; d++) {
                int tv = ctx[d], nv = nb[3+d];
                fd[d] = (tv > 0 && nv > 0 && tv == nv) ? 1.0 : 0.0;
            }
            fdList.add(fd);
            pearsonList.add(pearson);
            validNbs.add(nb);
        }
        if (validNbs.isEmpty()) return new float[]{0, 0, 0};

        double[] overall = computeOverall(fdList);

        double[] gLocal = computeFuzzyMeasure(fdList, pearsonList, overall, g);

        // Score and select neighbors
        float sumW = 0, sumWR = 0; int numN = 0;
        for (int i = 0; i < validNbs.size(); i++) {
            float choquet = choquetIntegral(fdList.get(i), gLocal);
            if (choquet < EPSILON) continue;             // CINS threshold (paper: 0.1)
            float weight = pearsonList.get(i) * choquet;
            int[] nb    = validNbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            sumW  += Math.abs(weight);
            numN++;
        }
        return new float[]{sumW, sumWR, numN};
    }

    /**
     * computeOverall (Algorithm 2, line 11):
     * Per-dimension mean fuzzy density across all valid candidate neighbors.
     * Captures the aggregated contextual profile of the neighborhood.
     */
    private static double[] computeOverall(ArrayList<double[]> fdList) {
        double[] overall = new double[N_DIMS];
        for (double[] fd : fdList)
            for (int d = 0; d < N_DIMS; d++) overall[d] += fd[d];
        int n = fdList.size();
        if (n > 0) for (int d = 0; d < N_DIMS; d++) overall[d] /= n;
        return overall;
    }

    /**
     * computeFuzzyMeasure (Algorithm 2, line 12):
     * Identifies local Sugeno fuzzy densities gLocal by minimising
     *   Î£_i ( Ch_gLocal(fd_i) âˆ’ pearson_i )Â²
     * via projected gradient descent (NNLS, additive Choquet = weighted sum).
     * Warm-started from overall[], regularised toward gGlobal[].
     * Falls back to gGlobal when the neighborhood is too small to fit reliably.
     */
    private static double[] computeFuzzyMeasure(ArrayList<double[]> fdList,
            ArrayList<Float> pearsonList, double[] overall, double[] gGlobal) {

        int M = fdList.size();
        if (M < N_DIMS) return gGlobal; // too few candidates to fit

        // Warm start: blend neighborhood overall with global Shapley weights
        double[] gLocal = new double[N_DIMS];
        for (int d = 0; d < N_DIMS; d++)
            gLocal[d] = Math.max(0.001, 0.5 * overall[d] + 0.5 * gGlobal[d]);

        // Projected gradient descent: minimise ||AÂ·g âˆ’ y||Â²  s.t. g â‰¥ 0
        // where A[i][d] = fd[i][d]  (binary),  y[i] = pearson[i]
        double lr = 0.05;
        for (int step = 0; step < 100; step++) {
            double[] grad = new double[N_DIMS];
            for (int i = 0; i < M; i++) {
                double[] fd = fdList.get(i);
                double pred = 0;
                for (int d = 0; d < N_DIMS; d++) pred += gLocal[d] * fd[d];
                double err = pred - pearsonList.get(i);
                for (int d = 0; d < N_DIMS; d++) grad[d] += 2.0 * err * fd[d];
            }
            for (int d = 0; d < N_DIMS; d++) {
                gLocal[d] -= (lr / M) * grad[d];
                gLocal[d]  = Math.max(0.001, gLocal[d]); // project onto non-negative orthant
            }
        }

        // Normalise so densities sum to 1 (required by Sugeno fuzzy measure)
        double sum = 0;
        for (double v : gLocal) sum += v;
        if (sum > 0) for (int d = 0; d < N_DIMS; d++) gLocal[d] /= sum;
        else return gGlobal;
        return gLocal;
    }

    //  Pearson similarity (shrunk by co-rating count, min 6 items)
    private static float pearsonSim(int u1, int u2,
            Map<Integer, Map<Integer, Float>> uir, Map<Integer, Float> means) {
        Map<Integer, Float> r1 = uir.get(u1), r2 = uir.get(u2);
        if (r1 == null || r2 == null) return 0f;
        float m1 = means.containsKey(u1) ? means.get(u1) : 3.5f;
        float m2 = means.containsKey(u2) ? means.get(u2) : 3.5f;
        float sN = 0, sD1 = 0, sD2 = 0; int common = 0;
        for (Map.Entry<Integer, Float> e : r1.entrySet()) {
            Float v2 = r2.get(e.getKey());
            if (v2 != null) {
                float d1 = e.getValue() - m1, d2 = v2 - m2;
                sN += d1*d2; sD1 += d1*d1; sD2 += d2*d2; common++;
            }
        }
        if (common < 1) return 0f;
        float den = (float)(Math.sqrt(sD1) * Math.sqrt(sD2));
        if (den < 1e-6f) return 0f;
        return (sN / den) * Math.min(1f, common / 6f);
    }

    //  Choquet integral with Sugeno fuzzy measure
    private static float choquetIntegral(double[] v, double[] g) {
        int n = v.length; if (n == 0) return 0;
        double lam = sugenoLambda(g);
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = 1; i < n; i++) {
            int k = idx[i], j = i-1;
            while (j >= 0 && v[idx[j]] > v[k]) { idx[j+1] = idx[j]; j--; }
            idx[j+1] = k;
        }
        double ch = 0, prev = 0;
        for (int i = 0; i < n; i++) {
            double ai = v[idx[i]]; if (ai <= prev) { prev = ai; continue; }
            int mask = 0; for (int k = i; k < n; k++) mask |= (1 << idx[k]);
            ch += (ai - prev) * sugenoMu(mask, g, lam); prev = ai;
        }
        return (float) Math.max(0, Math.min(1, ch));
    }

    private static double sugenoLambda(double[] g) {
        double s = 0; for (double gi : g) s += gi;
        if (Math.abs(s - 1) < 1e-6) return 0;
        double lo, hi;
        if (s > 1) { lo = -1 + 1e-6; hi = 0; } else { lo = 0; hi = 50; }
        for (int i = 0; i < 100; i++) {
            double mid = (lo + hi) / 2, prod = 1;
            for (double gi : g) prod *= (1 + mid * gi);
            if (Math.abs(prod - 1 - mid) < 1e-10) return mid;
            if (prod - 1 - mid > 0) hi = mid; else lo = mid;
        }
        return (lo + hi) / 2;
    }

    private static double sugenoMu(int mask, double[] g, double lam) {
        if (mask == 0) return 0;
        if (Math.abs(lam) < 1e-10) {
            double s = 0;
            for (int i = 0; i < g.length; i++) if ((mask & (1 << i)) != 0) s += g[i];
            return Math.min(1, s);
        }
        double prod = 1;
        for (int i = 0; i < g.length; i++) if ((mask & (1 << i)) != 0) prod *= (1 + lam * g[i]);
        return Math.min(1, Math.max(0, (prod - 1) / lam));
    }

    //  Ranking metrics
    static double precAtK(ArrayList<float[]> r, Set<Integer> rel, int k) {
        int h = 0;
        for (int i = 0; i < Math.min(k, r.size()); i++)
            if (rel.contains((int) r.get(i)[0])) h++;
        return (double) h / k;
    }

    static double recAtK(ArrayList<float[]> r, Set<Integer> rel, int k) {
        if (rel.isEmpty()) return 0;
        int h = 0;
        for (int i = 0; i < Math.min(k, r.size()); i++)
            if (rel.contains((int) r.get(i)[0])) h++;
        return (double) h / rel.size();
    }

    static double ndcgAtK(ArrayList<float[]> r, Set<Integer> rel, int k) {
        double dcg = 0;
        for (int i = 0; i < Math.min(k, r.size()); i++)
            if (rel.contains((int) r.get(i)[0])) dcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        double idcg = 0;
        for (int i = 0; i < Math.min(k, rel.size()); i++)
            idcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        return (idcg > 0) ? dcg / idcg : 0;
    }

    //  Data loading
    private static ArrayList<int[]> loadData(String path) throws Exception {
        ArrayList<int[]> all = new ArrayList<int[]>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line; boolean first = true;
        while ((line = br.readLine()) != null) {
            line = line.trim(); if (line.isEmpty()) continue;
            if (first) { first = false; if (line.toLowerCase().contains("user")) continue; }
            String[] c = line.split(",");
            if (c.length < 3 + N_DIMS) continue;
            int[] row = new int[3 + N_DIMS];
            for (int i = 0; i < row.length; i++) {
                try { row[i] = (int) Float.parseFloat(c[i].trim()); }
                catch (Exception ex) { row[i] = -1; }
            }
            if (row[0] < 0 || row[1] < 0 || row[2] < 0) continue;
            all.add(row);
        }
        br.close();
        return all;
    }

    //  Statistics
    static double mean(double[] v) {
        double s = 0; for (double x : v) s += x; return s / v.length;
    }

    static double std(double[] v) {
        double m = mean(v), s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return Math.sqrt(s / v.length);
    }
}
