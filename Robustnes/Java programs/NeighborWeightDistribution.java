import java.io.*;
import java.util.*;

/**
 * FO-CAR CINS -- Final Choquet-based neighbor weight distribution, genuine
 * vs. injected.
 *
 * For a sample of test predictions (one representative train/test split --
 * this is for a distribution plot, not a precision statistic, so no
 * multi-fold averaging is needed), builds the real candidate pool + a 50%
 * irrelevant-neighbor injection (same method as IrrelevantNeighborInjection),
 * computes each candidate's final CINS weight (pearson * choquet, the value
 * actually used in the weighted-average prediction, including candidates
 * that fall below EPSILON so the cutoff's effect on the shape is visible),
 * and writes one row per candidate: group (genuine/injected), weight.
 *
 * SAMPLE_PREDICTIONS caps how many test rows are processed, to keep the
 * output file a reasonable size for a histogram/violin plot (each prediction
 * contributes ~1.5x its real-candidate count in rows, real + injected).
 */
public class NeighborWeightDistribution {

    static String DATA_CSV = "RichMTV_clean.csv";
    static int    N_DIMS   = 8;

    static int    SHAPLEY_ITER    = 30;
    static int    SHAPLEY_SAMPLES = 200;
    static double EPSILON_1       = 0.0;
    static double EPSILON         = 0.1; // paper default

    static double INJECTION_RATE = 0.5;

    static long SEED = 1;
    static int  K_FOLDS = 3; // only fold 1 of this split is used

    static int SAMPLE_PREDICTIONS = 5000; // cap on test rows processed
    static double SUBSAMPLE_FRACTION = 1.0; // <1.0 for a fast smoke test

    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR CINS -- Neighbor Weight Distribution (genuine vs injected) ===\n");
        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        if (SUBSAMPLE_FRACTION < 1.0) {
            Collections.shuffle(allRows, new Random(0));
            int keep = (int) Math.round(allRows.size() * SUBSAMPLE_FRACTION);
            allRows = new ArrayList<int[]>(allRows.subList(0, keep));
            System.out.println("Subsampled to: " + allRows.size() + " rows (smoke test)");
        }

        ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
        Collections.shuffle(shuffled, new Random(SEED));
        int fs = shuffled.size() / K_FOLDS;
        ArrayList<int[]> test  = new ArrayList<int[]>(shuffled.subList(0, fs));
        ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(fs, shuffled.size()));
        System.out.println("Train: " + train.size() + " rows, Test: " + test.size() + " rows");

        Random rng = new Random(SEED * 1000L + 1);
        if (test.size() > SAMPLE_PREDICTIONS) {
            Collections.shuffle(test, rng);
            test = new ArrayList<int[]>(test.subList(0, SAMPLE_PREDICTIONS));
        }
        System.out.println("Sampled " + test.size() + " test predictions for weight collection");

        // -- Build the same fold context as the other experiments --
        Map<Long, float[]> pairAcc = new HashMap<Long, float[]>();
        ArrayList<int[]> allForUir = new ArrayList<int[]>(train);
        allForUir.addAll(test);
        for (int[] s : allForUir) {
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

        Map<Integer, ArrayList<int[]>> itemIdx = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] r : train) {
            ArrayList<int[]> l = itemIdx.get(r[1]);
            if (l == null) { l = new ArrayList<int[]>(); itemIdx.put(r[1], l); }
            l.add(r);
        }

        System.out.println("Computing Shapley weights...");
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

        PrintWriter pw = new PrintWriter(new FileWriter("neighbor_weight_distribution.csv"));
        pw.println("group,pearson,choquet,final_weight,kept");

        long nGenuine = 0, nInjected = 0;
        int processed = 0;

        for (int[] tst : test) {
            int uid = tst[0];
            float uMean = uMeans.containsKey(uid) ? uMeans.get(uid) : gMean;
            int[] ctx = new int[N_DIMS];
            Arrays.fill(ctx, -1);
            for (int d = 0; d < N_DIMS; d++) if (3 + d < tst.length) ctx[d] = tst[3 + d];

            ArrayList<int[]> candidates = step2_generateNeighborhood(tst[1], ctx, g, itemIdx);

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
                fdList.add(fd); pearsonList.add(pearson); validNbs.add(nb);
            }
            if (validNbs.isEmpty()) { processed++; continue; }

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

            double[] overall = computeOverall(fdAll);
            double[] gLocal  = computeFuzzyMeasure(fdAll, pearsonAll, overall, g);

            for (int i = 0; i < nbAll.size(); i++) {
                float choquet = choquetIntegral(fdAll.get(i), gLocal);
                float pearson = pearsonAll.get(i);
                float finalWeight = pearson * choquet;
                boolean kept = choquet >= EPSILON;
                String group = (i < validNbs.size()) ? "genuine" : "injected";
                pw.printf(Locale.US, "%s,%.4f,%.4f,%.4f,%d%n", group, pearson, choquet, finalWeight, kept ? 1 : 0);
                if (i < validNbs.size()) nGenuine++; else nInjected++;
            }

            processed++;
            if (processed % 1000 == 0) System.out.println("  processed " + processed + "/" + test.size());
        }

        pw.close();
        System.out.println("\nWrote " + (nGenuine + nInjected) + " candidate weight rows ("
            + nGenuine + " genuine, " + nInjected + " injected) to neighbor_weight_distribution.csv");
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

    // ---- Steps 1/2 + shared math, ported from RichMTV_Robustness_1.java ----

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
}
