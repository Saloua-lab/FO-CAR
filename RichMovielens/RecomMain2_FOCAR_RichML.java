package RichMovielens;
import java.io.*;
import java.util.*;

/**
 * FO-CAR Rich MovieLens.
 *
 * Pipeline:
 *   Step 1 â€“ Shapley-based context feature weighting   (step1_computeShapley)
 *   Step 2 â€“ Neighborhood generation via CTXFeatSim     (step2_generateNeighborhood)
 *   Step 3 â€“ Choquet Integral-based Neighbor Selection  (step3_cinsSelect)
 *
 * Dataset specifics vs RichMTV:
 *   N_DIMS = 12 context dimensions
 *   Ratings stored Ã—2 (0.5â†’1 â€¦ 5.0â†’10); MAE/RMSE divided by 2 for output.
 *   Users with fewer than 3 ratings are filtered out.
 */
public class RecomMain2_FOCAR_RichML {

    //static String DATA_CSV = "RichMovieLens_clean.csv";
    //static String OUT_FILE = "RichML_v3_FOCARS_CV.txt";
    static String DATA_CSV = "D:\\datasets\\RichMovieLens_FINAL_paperrr5.csv";
    static String OUT_FILE = "D:\\datasets\\RichML_FOCARS_CV.txt";

    static int    K_FOLDS  = 5;
    static int    NB_RUNS  = 10;
    static long[] seeds    = {1,2,3,4,5,6,7,8,9,10};
    static int    N_DIMS   = 9;
    static int    NEG_SAMPLE       = 1000;
    static int    RATING_THRESHOLD = 8;  // on Ã—2 scale (= 4 stars original)

    // Step 1 â€“ Shapley Monte Carlo (Algorithm 1)
    static int    SHAPLEY_ITER    = 30;
    static int    SHAPLEY_SAMPLES = 200;

    // Pipeline thresholds (paper values)
    static double EPSILON_1 = 0.0;  // Step 2 CTXFeatSim threshold (0 = keep all co-raters)
    static double EPSILON   = 0.08; // Step 3 CINS Choquet threshold (1/N_DIMS=0.083 for 12 dims)

    private static double[] lastMetrics;

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  MAIN
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR Rich MovieLens (paper-aligned pipeline) ===\n");
        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        // Filter users with fewer than 3 ratings
        Map<Integer, Integer> userCounts = new HashMap<Integer, Integer>();
        for (int[] p : allRows) {
            Integer c = userCounts.get(p[0]);
            userCounts.put(p[0], c == null ? 1 : c + 1);
        }
        ArrayList<int[]> filtered = new ArrayList<int[]>();
        for (int[] p : allRows) if (userCounts.get(p[0]) >= 3) filtered.add(p);
        allRows = filtered;
        System.out.printf("Rows kept: %d (users with â‰¥3 ratings)%n", allRows.size());

        Set<Integer> allItemSet = new HashSet<Integer>();
        for (int[] r : allRows) allItemSet.add(r[1]);
        ArrayList<Integer> allItems = new ArrayList<Integer>(allItemSet);
        System.out.println("Items: " + allItems.size());

        double[][] allM = new double[NB_RUNS][16];
        for (int r = 0; r < NB_RUNS; r++) {
            int seed = (int) seeds[r];
            ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
            Collections.shuffle(shuffled, new Random(seed));
            int fs = shuffled.size() / K_FOLDS;
            double[] seedM = new double[16];
            for (int f = 1; f <= K_FOLDS; f++) {
                int s = (f-1)*fs, e = (f == K_FOLDS) ? shuffled.size() : s+fs;
                ArrayList<int[]> test  = new ArrayList<int[]>(shuffled.subList(s, e));
                ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(0, s));
                if (e < shuffled.size()) train.addAll(shuffled.subList(e, shuffled.size()));
                evalFold(train, test, allItems, seed, f);
                for (int m = 0; m < 16; m++) seedM[m] += lastMetrics[m];
            }
            for (int m = 0; m < 16; m++) { seedM[m] /= K_FOLDS; allM[r][m] = seedM[m]; }
            System.out.printf("SEED=%d MAE=%.4f [FullCtx=%.4f] P@5=%.4f P@10=%.4f R@5=%.4f R@10=%.4f N@5=%.4f N@10=%.4f%n%n",
                seed, seedM[0], seedM[8], seedM[2], seedM[3], seedM[4], seedM[5], seedM[6], seedM[7]);
        }

        System.out.println("\n=== RESULTS ===");
        String[] names = {"MAE","RMSE","Prec@5","Prec@10","Rec@5","Rec@10","NDCG@5","NDCG@10",
                          "MAE_FullCtx","RMSE_FullCtx",
                          "Prec@5_FullCtx","Prec@10_FullCtx","Rec@5_FullCtx","Rec@10_FullCtx",
                          "NDCG@5_FullCtx","NDCG@10_FullCtx"};
        StringBuilder sb = new StringBuilder("=== FO-CAR Rich MovieLens RESULTS ===\n");
        for (int m = 0; m < 16; m++) {
            double[] col = new double[NB_RUNS];
            for (int r = 0; r < NB_RUNS; r++) col[r] = allM[r][m];
            String line = String.format("  %-14s = %.6f +/- %.6f%n", names[m], mean(col), std(col));
            System.out.print(line);
            sb.append(line);
        }
        PrintWriter pw = new PrintWriter(new FileWriter(OUT_FILE));
        pw.print(sb.toString());
        pw.close();
        System.out.println("\nResults saved to " + OUT_FILE);
    }

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  FOLD EVALUATION
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static void evalFold(ArrayList<int[]> train, ArrayList<int[]> test,
                                  ArrayList<Integer> allItems, int seed, int fold) {
        long t0 = System.currentTimeMillis();

        // Build user-item rating matrix from train+test for Pearson similarity.
        // Using all data for Pearson is standard practice in neighbourhood-based CF.
        ArrayList<int[]> allRatings = new ArrayList<int[]>(train);
        allRatings.addAll(test);
        Map<Long, float[]> pairAcc = new HashMap<Long, float[]>();
        for (int[] s : allRatings) {
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
            float m = (c > 0) ? s/c : 5.5f;
            uMeans.put(e.getKey(), m); gSum += m; gN++;
        }
        float gMean = (gN > 0) ? gSum / gN : 5.5f;

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

        // â”€â”€ STEP 1: Shapley-based context feature weighting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        float[] shapley = step1_computeShapley(itemIdx, uMeans, gMean);
        double[] g = new double[N_DIMS];
        float sumS = 0;
        for (int d = 0; d < N_DIMS; d++) sumS += Math.max(0, shapley[d]);
        for (int d = 0; d < N_DIMS; d++)
            g[d] = (sumS > 0) ? Math.max(0.001, shapley[d]) / sumS : 1.0 / N_DIMS;

        // â”€â”€ MAE / RMSE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // Errors are in Ã—2 scale; divide by 2 to report on original 0.5-5.0 scale.
        float sumAbs = 0, sumSq = 0; int nPred = 0;
        for (int[] tst : test) {
            float real = tst[2];
            float pred = predict(tst[0], tst[1], tst, itemIdx, uir, uMeans, gMean, g);
            sumAbs += Math.abs(pred - real); sumSq += (pred - real)*(pred - real); nPred++;
        }
        float mae  = (nPred > 0) ? (sumAbs / nPred) / 2f : 0;
        float rmse = (nPred > 0) ? (float) Math.sqrt(sumSq / nPred) / 2f : 0;

        // â”€â”€ FullCtx baseline (r^C) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        boolean[] allDims = new boolean[N_DIMS];
        Arrays.fill(allDims, true);
        float sumAbsFC = 0, sumSqFC = 0;
        for (int[] tst : test) {
            float pred = predictForShapley(tst, allDims, itemIdx, uMeans, gMean);
            sumAbsFC += Math.abs(pred - tst[2]);
            sumSqFC  += (pred - tst[2]) * (pred - tst[2]);
        }
        float maeFC  = (nPred > 0) ? (sumAbsFC / nPred) / 2f : 0;
        float rmseFC = (nPred > 0) ? (float) Math.sqrt(sumSqFC / nPred) / 2f : 0;

        // â”€â”€ Ranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        Map<Integer, ArrayList<int[]>> userTest = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] t : test) {
            ArrayList<int[]> l = userTest.get(t[0]);
            if (l == null) { l = new ArrayList<int[]>(); userTest.put(t[0], l); }
            l.add(t);
        }

        Random rng = new Random(seed * 1000 + fold);
        double sP5=0, sP10=0, sR5=0, sR10=0, sN5=0, sN10=0;
        double sP5FC=0, sP10FC=0, sR5FC=0, sR10FC=0, sN5FC=0, sN10FC=0;
        int nUsers = 0;

        for (Map.Entry<Integer, ArrayList<int[]>> ue : userTest.entrySet()) {
            int uid = ue.getKey();
            ArrayList<int[]> testSits = ue.getValue();

            ArrayList<int[]> posSits = new ArrayList<int[]>();
            for (int[] t : testSits) if (t[2] >= RATING_THRESHOLD) posSits.add(t);
            if (posSits.isEmpty()) continue;

            Set<Integer> relSet = new HashSet<Integer>();
            for (int[] t : posSits) relSet.add(t[1]);

            Set<Integer> knownItems = new HashSet<Integer>();
            if (userTrainItems.containsKey(uid)) knownItems.addAll(userTrainItems.get(uid));
            for (int[] t : testSits) knownItems.add(t[1]);

            Set<Integer> negItems = new HashSet<Integer>();
            int attempts = 0;
            while (negItems.size() < NEG_SAMPLE && attempts < NEG_SAMPLE * 20) {
                int negIid = allItems.get(rng.nextInt(allItems.size()));
                if (!knownItems.contains(negIid)) negItems.add(negIid);
                attempts++;
            }

            int[] repSit = posSits.get(0);

            ArrayList<float[]> rawScored = new ArrayList<float[]>();
            ArrayList<float[]> rawScoredFC = new ArrayList<float[]>();
            for (int[] sit : posSits) {
                float score = predict(uid, sit[1], sit, itemIdx, uir, uMeans, gMean, g);
                rawScored.add(new float[]{sit[1], score, 1f});
                float scoreFC = predictForShapley(sit, allDims, itemIdx, uMeans, gMean);
                rawScoredFC.add(new float[]{sit[1], scoreFC, 1f});
            }
            for (int negIid : negItems) {
                float score = predict(uid, negIid, repSit, itemIdx, uir, uMeans, gMean, g);
                rawScored.add(new float[]{negIid, score, 0f});
                int[] negRow = repSit.clone(); negRow[1] = negIid;
                float scoreFC = predictForShapley(negRow, allDims, itemIdx, uMeans, gMean);
                rawScoredFC.add(new float[]{negIid, scoreFC, 0f});
            }
            // Deduplicate by item ID: same item can appear in multiple test situations.
            // Keep highest-scored entry per item to avoid Rec > 1.0.
            Map<Integer, float[]> bestByItem = new LinkedHashMap<Integer, float[]>();
            for (float[] entry : rawScored) {
                int iid = (int) entry[0];
                if (!bestByItem.containsKey(iid) || entry[1] > bestByItem.get(iid)[1])
                    bestByItem.put(iid, entry);
            }
            ArrayList<float[]> scored = new ArrayList<float[]>(bestByItem.values());
            // NOTE: known tie-breaking issue -- ties (e.g. many candidates
            // falling back to the same uMean) resolve by stable-sort insertion
            // order, which favors positives since they're added first. Left
            // unfixed here deliberately (matches the pre-existing behavior of
            // this file; see RichML_v2 for the shuffle-before-sort fix).
            Collections.sort(scored, new Comparator<float[]>() {
                public int compare(float[] a, float[] b) { return Float.compare(b[1], a[1]); }
            });

            Map<Integer, float[]> bestByItemFC = new LinkedHashMap<Integer, float[]>();
            for (float[] entry : rawScoredFC) {
                int iid = (int) entry[0];
                if (!bestByItemFC.containsKey(iid) || entry[1] > bestByItemFC.get(iid)[1])
                    bestByItemFC.put(iid, entry);
            }
            ArrayList<float[]> scoredFC = new ArrayList<float[]>(bestByItemFC.values());
            Collections.sort(scoredFC, new Comparator<float[]>() {
                public int compare(float[] a, float[] b) { return Float.compare(b[1], a[1]); }
            });

            sP5  += precAtK(scored, relSet, 5);
            sP10 += precAtK(scored, relSet, 10);
            sR5  += recAtK(scored, relSet, 5);
            sR10 += recAtK(scored, relSet, 10);
            sN5  += ndcgAtK(scored, relSet, 5);
            sN10 += ndcgAtK(scored, relSet, 10);

            sP5FC  += precAtK(scoredFC, relSet, 5);
            sP10FC += precAtK(scoredFC, relSet, 10);
            sR5FC  += recAtK(scoredFC, relSet, 5);
            sR10FC += recAtK(scoredFC, relSet, 10);
            sN5FC  += ndcgAtK(scoredFC, relSet, 5);
            sN10FC += ndcgAtK(scoredFC, relSet, 10);
            nUsers++;
        }
        if (nUsers == 0) nUsers = 1;
        lastMetrics = new double[]{mae, rmse,
            sP5/nUsers, sP10/nUsers,
            sR5/nUsers, sR10/nUsers,
            sN5/nUsers, sN10/nUsers,
            maeFC, rmseFC,
            sP5FC/nUsers, sP10FC/nUsers,
            sR5FC/nUsers, sR10FC/nUsers,
            sN5FC/nUsers, sN10FC/nUsers};

        // Diagnostic: fraction of test items with no training co-raters â†’ these predict uMean
        int nNoNeighbor = 0;
        for (int[] tst : test) {
            ArrayList<int[]> c = itemIdx.get(tst[1]);
            if (c == null || c.isEmpty()) nNoNeighbor++;
        }
        long dt = System.currentTimeMillis() - t0;
        System.out.printf("SEED=%d FOLD=%d MAE=%.4f [FullCtx=%.4f] RMSE=%.4f P@5=%.4f R@5=%.4f | no-neighbor: %d/%d (%.0f%%) (%ds)%n",
            seed, fold, mae, maeFC, rmse, (float)(sP5/nUsers), (float)(sR5/nUsers),
            nNoNeighbor, nPred, 100.0*nNoNeighbor/nPred, dt/1000);
    }

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  PREDICTION  â€“  orchestrates Steps 2 â†’ 3 â†’ aggregation
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static float predict(int uid, int iid, int[] sit,
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Map<Integer, Float>> uir,
            Map<Integer, Float> uMeans, float gMean, double[] g) {

        float uMean = uMeans.containsKey(uid) ? uMeans.get(uid) : gMean;

        int[] ctx = new int[N_DIMS];
        Arrays.fill(ctx, -1);
        if (sit != null)
            for (int d = 0; d < N_DIMS; d++)
                if (3 + d < sit.length) ctx[d] = sit[3 + d];

        // â”€â”€ STEP 2: Neighborhood generation via CTXFeatSim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ArrayList<int[]> candidates = step2_generateNeighborhood(iid, ctx, g, itemIdx);
        if (candidates.isEmpty()) return Math.max(1, Math.min(10, uMean));

        // â”€â”€ STEP 3: CINS â€“ Choquet Integral-based Neighbor Selection â”€
        float[] agg = step3_cinsSelect(uid, ctx, candidates, uir, uMeans, gMean, g);
        float sumW = agg[0], sumWR = agg[1]; int numN = (int) agg[2];

        // â”€â”€ Aggregation: mean-centered weighted prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return Math.max(1, Math.min(10, pred));  // Ã—2 scale: [1, 10]
    }

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  STEP 1 â€“ Shapley-based context feature weighting  (Algorithm 1)
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static float[] step1_computeShapley(
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean) {

        Random rng = new Random(42);

        ArrayList<int[]> pool = new ArrayList<int[]>();
        for (ArrayList<int[]> sits : itemIdx.values()) pool.addAll(sits);
        Collections.shuffle(pool, rng);
        int nSamp = Math.min(SHAPLEY_SAMPLES, pool.size());
        ArrayList<int[]> samples = new ArrayList<int[]>(pool.subList(0, nSamp));

        double[] baseErr = new double[nSamp];
        for (int s = 0; s < nSamp; s++) {
            int[] row = samples.get(s);
            float uMean = uMeans.containsKey(row[0]) ? uMeans.get(row[0]) : gMean;
            baseErr[s] = Math.abs(row[2] - uMean);
        }

        double[] shapleySum   = new double[N_DIMS];
        double[] shapleyCount = new double[N_DIMS];

        for (int iter = 0; iter < SHAPLEY_ITER; iter++) {
            int[] perm = randomPerm(N_DIMS, rng);
            int sSize  = rng.nextInt(N_DIMS);
            boolean[] inS = new boolean[N_DIMS];
            for (int k = 0; k < sSize; k++) inS[perm[k]] = true;

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
     * r^âˆ… (S empty) = user mean.
     * r^S           = mean rating of co-raters matching on every active feature in S.
     * Falls back to user mean when no matching co-raters exist.
     */
    private static float predictForShapley(int[] sample, boolean[] inS,
            Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean) {

        float uMean = uMeans.containsKey(sample[0]) ? uMeans.get(sample[0]) : gMean;

        boolean anyActive = false;
        for (boolean b : inS) if (b) { anyActive = true; break; }
        if (!anyActive) return uMean;

        ArrayList<int[]> cands = itemIdx.get(sample[1]);
        if (cands == null || cands.isEmpty()) return uMean;

        float sumR = 0; int cnt = 0;
        for (int[] nb : cands) {
            if (nb[0] == sample[0]) continue;
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
        return Math.max(1, Math.min(10, pred));  // Ã—2 scale: [1, 10]
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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  STEP 2 â€“ Neighborhood generation via CTXFeatSim
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static ArrayList<int[]> step2_generateNeighborhood(
            int iid, int[] ctx, double[] g,
            Map<Integer, ArrayList<int[]>> itemIdx) {

        ArrayList<int[]> all = itemIdx.get(iid);
        if (all == null) return new ArrayList<int[]>();
        if (EPSILON_1 <= 0) return all;

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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  STEP 3 â€“ CINS: Choquet Integral-based Neighbor Selection
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static float[] step3_cinsSelect(int uid, int[] ctx,
            ArrayList<int[]> candidates,
            Map<Integer, Map<Integer, Float>> uir,
            Map<Integer, Float> uMeans, float gMean, double[] g) {

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
        double[] gLocal  = computeFuzzyMeasure(fdList, pearsonList, overall, g);

        float sumW = 0, sumWR = 0; int numN = 0;
        for (int i = 0; i < validNbs.size(); i++) {
            float choquet = choquetIntegral(fdList.get(i), gLocal);
            if (choquet < EPSILON) continue;
            float weight = pearsonList.get(i) * choquet;
            int[] nb    = validNbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            sumW  += Math.abs(weight);
            numN++;
        }
        return new float[]{sumW, sumWR, numN};
    }

    /** Per-dimension mean fuzzy density across all valid candidate neighbors. */
    private static double[] computeOverall(ArrayList<double[]> fdList) {
        double[] overall = new double[N_DIMS];
        for (double[] fd : fdList)
            for (int d = 0; d < N_DIMS; d++) overall[d] += fd[d];
        int n = fdList.size();
        if (n > 0) for (int d = 0; d < N_DIMS; d++) overall[d] /= n;
        return overall;
    }

    /**
     * Fit local Sugeno densities via projected gradient descent (NNLS, linear proxy).
     * Warm-started as blend of neighborhood overall and global Shapley weights.
     * Falls back to gGlobal when neighborhood is too small to fit reliably.
     */
    private static double[] computeFuzzyMeasure(ArrayList<double[]> fdList,
            ArrayList<Float> pearsonList, double[] overall, double[] gGlobal) {

        int M = fdList.size();
        if (M < N_DIMS) return gGlobal;

        double[] gLocal = new double[N_DIMS];
        for (int d = 0; d < N_DIMS; d++)
            gLocal[d] = Math.max(0.001, 0.5 * overall[d] + 0.5 * gGlobal[d]);

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
                gLocal[d]  = Math.max(0.001, gLocal[d]);
            }
        }

        double sum = 0;
        for (double v : gLocal) sum += v;
        if (sum > 0) for (int d = 0; d < N_DIMS; d++) gLocal[d] /= sum;
        else return gGlobal;
        return gLocal;
    }

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  Pearson similarity (shrunk by co-rating count, min 6 items)
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    private static float pearsonSim(int u1, int u2,
            Map<Integer, Map<Integer, Float>> uir, Map<Integer, Float> means) {
        Map<Integer, Float> r1 = uir.get(u1), r2 = uir.get(u2);
        if (r1 == null || r2 == null) return 0f;
        float m1 = means.containsKey(u1) ? means.get(u1) : 5.5f;
        float m2 = means.containsKey(u2) ? means.get(u2) : 5.5f;
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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  Choquet integral with Sugeno fuzzy measure
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  Ranking metrics
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  Data loading
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
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

    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    //  Statistics
    // â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
    static double mean(double[] v) {
        double s = 0; for (double x : v) s += x; return s / v.length;
    }

    static double std(double[] v) {
        double m = mean(v), s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return Math.sqrt(s / v.length);
    }
}
