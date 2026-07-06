import java.io.*;
import java.util.*;

/**
 * FO-CAR CINS -- Irrelevant-Neighbor Injection Experiment, ported to the
 * LDOS-CoMoDa dataset (matches RecomMain2_FOCAR_LDOS_v1.java's pipeline:
 * N_DIMS=5, context columns {7,10,11,12,15} = time/location/weather/social/
 * mood, default mean 3.0). Same injection design as
 * IrrelevantNeighborInjection.java for RichMTV: real alternate (rating,
 * context) rows borrowed from the same underlying user's own history.
 *
 * Data note: this repo's LDOS-CoMoDa.csv is semicolon-separated (unlike
 * RecomMain2_FOCAR_LDOS_v1.java's original "LDOS-CoMoDa2.csv", which expects
 * commas) -- loadData() here splits on ";" to match the file actually
 * present in CARSJava/.
 */
public class IrrelevantNeighborInjection_LDOS {

    static String DATA_CSV = "LDOS-CoMoDa.csv";
    static int[]  CTX_COLS = {7, 10, 11, 12, 15}; // time, location, weather, social, mood
    static int    N_DIMS   = CTX_COLS.length;

    static int    SHAPLEY_ITER    = 30;
    static int    SHAPLEY_SAMPLES = 200;
    static double EPSILON_1       = 0.0; // Step 2 CTXFeatSim threshold (paper: keep all co-raters)
    static double EPSILON         = 0.1; // Step 3 CINS Choquet threshold (paper default)

    static double INJECTION_RATE = 0.5; // irrelevant duplicates injected, relative to real pool size
    static float  DEFAULT_MEAN   = 3.0f; // matches RecomMain2_FOCAR_LDOS_v1's convention

    static int    RUNS    = 10; // dataset is tiny (~2300 rows) -- full budget runs in seconds
    static int    K_FOLDS = 5;
    static long[] seeds   = {1,2,3,4,5,6,7,8,9,10};

    static double SUBSAMPLE_FRACTION = 1.0;

    private static long statInjectedTotal      = 0;
    private static long statInjectedEliminated = 0;
    private static double statWtotalNoCins = 0, statWinjNoCins = 0;
    private static double statWtotalCins   = 0, statWinjCins   = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR CINS (LDOS-CoMoDa) -- Irrelevant-Neighbor Injection Experiment ===\n");
        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        if (SUBSAMPLE_FRACTION < 1.0) {
            Collections.shuffle(allRows, new Random(0));
            int keep = (int) Math.round(allRows.size() * SUBSAMPLE_FRACTION);
            allRows = new ArrayList<int[]>(allRows.subList(0, keep));
            System.out.println("Subsampled to: " + allRows.size() + " rows (smoke test)");
        }

        double[][] allM = new double[RUNS][8];

        for (int r = 0; r < RUNS; r++) {
            int seed = (int) seeds[r % seeds.length];
            ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
            Collections.shuffle(shuffled, new Random(seed));
            int fs = shuffled.size() / K_FOLDS;

            double[] seedM = new double[8];
            for (int f = 1; f <= K_FOLDS; f++) {
                int s = (f - 1) * fs, e = (f == K_FOLDS) ? shuffled.size() : s + fs;
                ArrayList<int[]> test  = new ArrayList<int[]>(shuffled.subList(s, e));
                ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(0, s));
                if (e < shuffled.size()) train.addAll(shuffled.subList(e, shuffled.size()));

                double[] foldM = evalFold(train, test, seed, f);
                for (int k = 0; k < 8; k++) seedM[k] += foldM[k];
            }
            for (int k = 0; k < 8; k++) { seedM[k] /= K_FOLDS; allM[r][k] = seedM[k]; }
        }

        double[] avg = new double[8];
        for (int k = 0; k < 8; k++) {
            double s = 0; for (int r = 0; r < RUNS; r++) s += allM[r][k];
            avg[k] = s / RUNS;
        }

        System.out.println("\n=== RESULTS (avg over " + RUNS + " runs x " + K_FOLDS + " folds) ===");
        System.out.printf(Locale.US, "%-28s %-10s %-10s%n", "Condition / Arm", "MAE", "RMSE");
        System.out.printf(Locale.US, "%-28s %-10.4f %-10.4f%n", "No injection / No-CINS", avg[0], avg[1]);
        System.out.printf(Locale.US, "%-28s %-10.4f %-10.4f%n", "No injection / CINS",    avg[2], avg[3]);
        System.out.printf(Locale.US, "%-28s %-10.4f %-10.4f%n", "+50% irrelevant / No-CINS", avg[4], avg[5]);
        System.out.printf(Locale.US, "%-28s %-10.4f %-10.4f%n", "+50% irrelevant / CINS",    avg[6], avg[7]);

        double deltaNoCins = avg[4] - avg[0];
        double deltaCins   = avg[6] - avg[2];
        System.out.printf(Locale.US, "%nMAE degradation from injection: No-CINS +%.4f, CINS +%.4f (lower = more robust to irrelevant neighbors)%n",
            deltaNoCins, deltaCins);

        double elimRate = (statInjectedTotal > 0) ? 100.0 * statInjectedEliminated / statInjectedTotal : 0;
        System.out.printf(Locale.US, "Injected irrelevant neighbors eliminated by CINS (choquet<EPSILON=%.2f): %d/%d (%.1f%%)%n",
            EPSILON, statInjectedEliminated, statInjectedTotal, elimRate);

        double irNoCins = (statWtotalNoCins > 0) ? statWinjNoCins / statWtotalNoCins : 0;
        double irCins   = (statWtotalCins   > 0) ? statWinjCins   / statWtotalCins   : 0;
        System.out.printf(Locale.US, "%nInfluence Ratio (share of total prediction weight coming from injected neighbors):%n");
        System.out.printf(Locale.US, "  IR(No-CINS) = %.4f (%.2f%%)%n", irNoCins, irNoCins * 100);
        System.out.printf(Locale.US, "  IR(CINS)    = %.4f (%.2f%%)%n", irCins, irCins * 100);
        System.out.printf(Locale.US, "  IR(CINS) %s IR(No-CINS) -> CINS %s injected neighbors relative influence%n",
            (irCins < irNoCins) ? "<" : ">=",
            (irCins < irNoCins) ? "reduces" : "does NOT reduce");

        writeCsv(avg, deltaNoCins, deltaCins, elimRate, irNoCins, irCins, "irrelevant_neighbor_injection_LDOS_results.csv");
    }

    private static void writeCsv(double[] avg, double deltaNoCins, double deltaCins, double elimRate,
            double irNoCins, double irCins, String path) throws IOException {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("condition,arm,MAE,RMSE");
        pw.printf(Locale.US, "no_injection,NoCINS,%.6f,%.6f%n", avg[0], avg[1]);
        pw.printf(Locale.US, "no_injection,CINS,%.6f,%.6f%n", avg[2], avg[3]);
        pw.printf(Locale.US, "inject_50pct,NoCINS,%.6f,%.6f%n", avg[4], avg[5]);
        pw.printf(Locale.US, "inject_50pct,CINS,%.6f,%.6f%n", avg[6], avg[7]);
        pw.println();
        pw.printf(Locale.US, "MAE_degradation_NoCINS,%.6f%n", deltaNoCins);
        pw.printf(Locale.US, "MAE_degradation_CINS,%.6f%n", deltaCins);
        pw.printf(Locale.US, "injected_elimination_rate_pct,%.2f%n", elimRate);
        pw.printf(Locale.US, "influence_ratio_NoCINS,%.6f%n", irNoCins);
        pw.printf(Locale.US, "influence_ratio_CINS,%.6f%n", irCins);
        pw.close();
        System.out.println("\nWrote results to " + path);
    }

    private static double[] evalFold(ArrayList<int[]> train, ArrayList<int[]> test, int seed, int fold) {
        long t0 = System.currentTimeMillis();

        Map<Long, float[]> pairAcc = new HashMap<Long, float[]>();
        ArrayList<int[]> allRows = new ArrayList<int[]>(train);
        allRows.addAll(test);
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
            float m = (c > 0) ? s/c : DEFAULT_MEAN;
            uMeans.put(e.getKey(), m); gSum += m; gN++;
        }
        float gMean = (gN > 0) ? gSum / gN : DEFAULT_MEAN;

        Map<Integer, ArrayList<int[]>> itemIdx = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] r : train) {
            ArrayList<int[]> l = itemIdx.get(r[1]);
            if (l == null) { l = new ArrayList<int[]>(); itemIdx.put(r[1], l); }
            l.add(r);
        }

        float[] shapley = step1_computeShapley(itemIdx, uMeans, gMean);
        double[] g = new double[N_DIMS];
        float sumS = 0;
        for (int d = 0; d < N_DIMS; d++) sumS += Math.max(0, shapley[d]);
        for (int d = 0; d < N_DIMS; d++)
            g[d] = (sumS > 0) ? Math.max(0.001, shapley[d]) / sumS : 1.0 / N_DIMS;

        Set<Integer>[] dimSet = new HashSet[N_DIMS];
        for (int d = 0; d < N_DIMS; d++) dimSet[d] = new HashSet<Integer>();
        for (int[] r : train)
            for (int d = 0; d < N_DIMS; d++)
                if (r[3 + d] > 0) dimSet[d].add(r[3 + d]);
        ArrayList<Integer>[] dimValues = new ArrayList[N_DIMS];
        for (int d = 0; d < N_DIMS; d++) dimValues[d] = new ArrayList<Integer>(dimSet[d]);

        Map<Integer, ArrayList<int[]>> userRows = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] r : train) {
            ArrayList<int[]> l = userRows.get(r[0]);
            if (l == null) { l = new ArrayList<int[]>(); userRows.put(r[0], l); }
            l.add(r);
        }

        Random rng = new Random(seed * 1000L + fold);

        double[] sumAbs = new double[4], sumSq = new double[4];
        int nPred = 0;
        long foldInjectedTotal = 0, foldInjectedEliminated = 0;
        double foldWtotalNoCins = 0, foldWinjNoCins = 0, foldWtotalCins = 0, foldWinjCins = 0;

        for (int[] tst : test) {
            float uMean = uMeans.containsKey(tst[0]) ? uMeans.get(tst[0]) : gMean;
            int[] ctx = new int[N_DIMS];
            Arrays.fill(ctx, -1);
            for (int d = 0; d < N_DIMS; d++) if (3 + d < tst.length) ctx[d] = tst[3 + d];

            ArrayList<int[]> candidates = step2_generateNeighborhood(tst[1], ctx, g, itemIdx);
            float real = tst[2];

            float[] preds = new float[4];
            long[] injStats = new long[2];
            double[] weightStats = new double[4];
            predictWithInjection(tst[0], uMean, ctx, candidates, uir, uMeans, gMean, g, dimValues, userRows, rng, preds, injStats, weightStats);
            foldWtotalNoCins += weightStats[0]; foldWinjNoCins += weightStats[1];
            foldWtotalCins   += weightStats[2]; foldWinjCins   += weightStats[3];

            for (int k = 0; k < 4; k++) {
                float err = preds[k] - real;
                sumAbs[k] += Math.abs(err);
                sumSq[k]  += err * err;
            }
            nPred++;
            foldInjectedTotal      += injStats[0];
            foldInjectedEliminated += injStats[1];
        }

        statInjectedTotal      += foldInjectedTotal;
        statInjectedEliminated += foldInjectedEliminated;
        statWtotalNoCins += foldWtotalNoCins; statWinjNoCins += foldWinjNoCins;
        statWtotalCins   += foldWtotalCins;   statWinjCins   += foldWinjCins;

        double[] out = new double[8];
        for (int k = 0; k < 4; k++) {
            out[k*2+0] = (nPred > 0) ? sumAbs[k] / nPred : 0;
            out[k*2+1] = (nPred > 0) ? Math.sqrt(sumSq[k] / nPred) : 0;
        }

        long dt = System.currentTimeMillis() - t0;
        double elimPct = (foldInjectedTotal > 0) ? 100.0 * foldInjectedEliminated / foldInjectedTotal : 0;
        System.out.printf(Locale.US, "SEED=%d FOLD=%d MAE(noCINS/CINS no-inject)=%.4f/%.4f MAE(noCINS/CINS +inject)=%.4f/%.4f injected-eliminated=%d/%d (%.1f%%) (%dms)%n",
            seed, fold, out[0], out[2], out[4], out[6], foldInjectedEliminated, foldInjectedTotal, elimPct, dt);
        return out;
    }

    private static void predictWithInjection(int uid, float uMean, int[] ctx, ArrayList<int[]> candidates,
            Map<Integer, Map<Integer, Float>> uir, Map<Integer, Float> uMeans, float gMean, double[] g,
            ArrayList<Integer>[] dimValues, Map<Integer, ArrayList<int[]>> userRows, Random rng,
            float[] preds, long[] injStatsOut, double[] weightStatsOut) {

        injStatsOut[0] = 0; injStatsOut[1] = 0;

        if (candidates.isEmpty()) {
            float fb = clamp(uMean);
            Arrays.fill(preds, fb);
            return;
        }

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

        if (validNbs.isEmpty()) {
            float fb = clamp(uMean);
            Arrays.fill(preds, fb);
            return;
        }

        preds[0] = weightedPred(uMean, gMean, uMeans, validNbs, pearsonList);
        double[] overall0 = computeOverall(fdList);
        double[] gLocal0  = computeFuzzyMeasure(fdList, pearsonList, overall0, g);
        float[] choquet0  = new float[validNbs.size()];
        for (int i = 0; i < validNbs.size(); i++) choquet0[i] = choquetIntegral(fdList.get(i), gLocal0);
        preds[1] = weightedPredCINS(uMean, gMean, uMeans, validNbs, pearsonList, choquet0);

        int nInject = (int) Math.round(INJECTION_RATE * validNbs.size());
        ArrayList<Integer> order = new ArrayList<Integer>();
        for (int i = 0; i < validNbs.size(); i++) order.add(i);
        Collections.shuffle(order, rng);

        ArrayList<double[]> fdAll      = new ArrayList<double[]>(fdList);
        ArrayList<Float>    pearsonAll = new ArrayList<Float>(pearsonList);
        ArrayList<int[]>    nbAll      = new ArrayList<int[]>(validNbs);

        for (int k = 0; k < nInject; k++) {
            int baseIdx = order.get(k % order.size());
            int[] baseNb = validNbs.get(baseIdx);
            int baseUid = baseNb[0];

            ArrayList<int[]> userHist = userRows.get(baseUid);
            int[] altRow = null;
            if (userHist != null && userHist.size() > 1) {
                for (int attempt = 0; attempt < 5 && altRow == null; attempt++) {
                    int[] cand = userHist.get(rng.nextInt(userHist.size()));
                    if (cand != baseNb) altRow = cand;
                }
            }

            int[] injNb;
            if (altRow != null) {
                injNb = new int[3 + N_DIMS];
                injNb[0] = baseUid;
                injNb[1] = baseNb[1];
                injNb[2] = altRow[2];
                for (int d = 0; d < N_DIMS; d++) injNb[3 + d] = altRow[3 + d];
            } else {
                injNb = baseNb.clone();
                for (int d = 0; d < N_DIMS; d++)
                    if (ctx[d] > 0 && rng.nextDouble() < 0.75)
                        injNb[3+d] = mismatchValue(d, ctx[d], dimValues, rng);
            }

            double[] fd = new double[N_DIMS];
            for (int d = 0; d < N_DIMS; d++) {
                int tv = ctx[d], nv = injNb[3+d];
                fd[d] = (tv > 0 && nv > 0 && tv == nv) ? 1.0 : 0.0;
            }
            fdAll.add(fd);
            pearsonAll.add(pearsonList.get(baseIdx));
            nbAll.add(injNb);
        }

        double[] wStatsNoCins = new double[2];
        preds[2] = weightedPredWithInfluence(uMean, gMean, uMeans, nbAll, pearsonAll, validNbs.size(), wStatsNoCins);

        double[] overallInj = computeOverall(fdAll);
        double[] gLocalInj  = computeFuzzyMeasure(fdAll, pearsonAll, overallInj, g);
        float[] choquetInj  = new float[nbAll.size()];
        for (int i = 0; i < nbAll.size(); i++) choquetInj[i] = choquetIntegral(fdAll.get(i), gLocalInj);

        double[] wStatsCins = new double[2];
        preds[3] = weightedPredCINSWithInfluence(uMean, gMean, uMeans, nbAll, pearsonAll, choquetInj, validNbs.size(), wStatsCins);

        long eliminated = 0;
        for (int i = validNbs.size(); i < nbAll.size(); i++)
            if (choquetInj[i] < EPSILON) eliminated++;
        injStatsOut[0] = nInject;
        injStatsOut[1] = eliminated;

        weightStatsOut[0] = wStatsNoCins[0]; weightStatsOut[1] = wStatsNoCins[1];
        weightStatsOut[2] = wStatsCins[0];   weightStatsOut[3] = wStatsCins[1];
    }

    private static int mismatchValue(int d, int targetVal, ArrayList<Integer>[] dimValues, Random rng) {
        ArrayList<Integer> pool = dimValues[d];
        if (pool.isEmpty()) return -1;
        if (pool.size() == 1) {
            int v = pool.get(0);
            return (v != targetVal) ? v : -1;
        }
        int v; int guard = 0;
        do { v = pool.get(rng.nextInt(pool.size())); guard++; } while (v == targetVal && guard < 50);
        return v;
    }

    private static float weightedPred(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons) {
        float sumW = 0, sumWR = 0; int numN = 0;
        for (int i = 0; i < nbs.size(); i++) {
            float weight = pearsons.get(i);
            int[] nb = nbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            sumW  += Math.abs(weight);
            numN++;
        }
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return clamp(pred);
    }

    private static float weightedPredCINS(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons, float[] choquet) {
        float sumW = 0, sumWR = 0; int numN = 0;
        for (int i = 0; i < nbs.size(); i++) {
            if (choquet[i] < EPSILON) continue;
            float weight = pearsons.get(i) * choquet[i];
            int[] nb = nbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            sumW  += Math.abs(weight);
            numN++;
        }
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return clamp(pred);
    }

    private static float weightedPredWithInfluence(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons, int injectStart, double[] weightStatsOut) {
        float sumW = 0, sumWR = 0; int numN = 0;
        double wTotal = 0, wInj = 0;
        for (int i = 0; i < nbs.size(); i++) {
            float weight = pearsons.get(i);
            int[] nb = nbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            double aw = Math.abs(weight);
            sumW  += aw;
            wTotal += aw;
            if (i >= injectStart) wInj += aw;
            numN++;
        }
        weightStatsOut[0] = wTotal;
        weightStatsOut[1] = wInj;
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return clamp(pred);
    }

    private static float weightedPredCINSWithInfluence(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons, float[] choquet, int injectStart, double[] weightStatsOut) {
        float sumW = 0, sumWR = 0; int numN = 0;
        double wTotal = 0, wInj = 0;
        for (int i = 0; i < nbs.size(); i++) {
            if (choquet[i] < EPSILON) continue;
            float weight = pearsons.get(i) * choquet[i];
            int[] nb = nbs.get(i);
            float nMean = uMeans.containsKey(nb[0]) ? uMeans.get(nb[0]) : gMean;
            sumWR += weight * (nb[2] - nMean);
            double aw = Math.abs(weight);
            sumW  += aw;
            wTotal += aw;
            if (i >= injectStart) wInj += aw;
            numN++;
        }
        weightStatsOut[0] = wTotal;
        weightStatsOut[1] = wInj;
        float pred = (numN > 0 && sumW > 0) ? uMean + sumWR / sumW : uMean;
        return clamp(pred);
    }

    private static float clamp(float v) { return Math.max(1, Math.min(5, v)); }

    // ---- Steps 1/2 + shared math, ported from RecomMain2_FOCAR_LDOS_v1.java (identical to RichMTV) ----

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
        return Math.max(1, Math.min(5, pred));
    }

    private static int[] randomPerm(int n, Random rng) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        for (int i = n-1; i > 0; i--) {
            int j = rng.nextInt(i+1); int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        return p;
    }

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

    private static double ctxFeatSim(int[] ctx, int[] nb, double[] g) {
        double num = 0, den = 0;
        for (int d = 0; d < N_DIMS; d++) {
            if (ctx[d] <= 0 || nb[3+d] <= 0) continue;
            den += g[d];
            if (ctx[d] == nb[3+d]) num += g[d];
        }
        return (den > 0) ? num / den : 0;
    }

    private static double[] computeOverall(ArrayList<double[]> fdList) {
        double[] overall = new double[N_DIMS];
        for (double[] fd : fdList)
            for (int d = 0; d < N_DIMS; d++) overall[d] += fd[d];
        int n = fdList.size();
        if (n > 0) for (int d = 0; d < N_DIMS; d++) overall[d] /= n;
        return overall;
    }

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

    private static float pearsonSim(int u1, int u2,
            Map<Integer, Map<Integer, Float>> uir, Map<Integer, Float> means) {
        Map<Integer, Float> r1 = uir.get(u1), r2 = uir.get(u2);
        if (r1 == null || r2 == null) return 0f;
        float m1 = means.containsKey(u1) ? means.get(u1) : DEFAULT_MEAN;
        float m2 = means.containsKey(u2) ? means.get(u2) : DEFAULT_MEAN;
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

    // ---- Data loading: semicolon-separated LDOS-CoMoDa.csv, extract CTX_COLS ----
    private static ArrayList<int[]> loadData(String path) throws Exception {
        ArrayList<int[]> all = new ArrayList<int[]>();
        int maxCol = CTX_COLS[CTX_COLS.length - 1];
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line; boolean first = true;
        while ((line = br.readLine()) != null) {
            line = line.trim(); if (line.isEmpty()) continue;
            if (first) { first = false; if (line.toLowerCase().contains("user")) continue; }
            String[] c = line.split(";", -1);
            if (c.length <= maxCol) continue;
            int uid = parseIntSafe(c[0].trim());
            int iid = parseIntSafe(c[1].trim());
            int rat = parseIntSafe(c[2].trim());
            if (uid < 0 || iid < 0 || rat < 1 || rat > 5) continue;
            int[] row = new int[3 + N_DIMS];
            row[0] = uid; row[1] = iid; row[2] = rat;
            for (int d = 0; d < N_DIMS; d++)
                row[3 + d] = parseIntSafe(c[CTX_COLS[d]].trim());
            all.add(row);
        }
        br.close();
        return all;
    }

    private static int parseIntSafe(String s) {
        if (s == null || s.isEmpty()) return -1;
        try { return Integer.parseInt(s); } catch (Exception e) { return -1; }
    }
}
