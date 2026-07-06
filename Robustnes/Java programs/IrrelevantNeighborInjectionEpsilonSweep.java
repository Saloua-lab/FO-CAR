import java.io.*;
import java.util.*;

/**
 * FO-CAR CINS -- Irrelevant-Neighbor Injection, swept across EPSILON.
 *
 * Same injection design as IrrelevantNeighborInjection.java (real alternate
 * (rating, context) rows borrowed from the same underlying user, giving
 * naturally-varying context overlap rather than a synthetic total mismatch),
 * but instead of a single fixed EPSILON=0.1, reports MAE, elimination rate,
 * and Influence Ratio (IR = injected-neighbor weight share / total weight)
 * at every threshold in EPSILONS. Answers: how much would a stricter CINS
 * threshold actually cost in accuracy to buy a stronger influence-reduction
 * effect, so an honest operating point can be chosen instead of reporting
 * only the paper's default.
 *
 * No-CINS (plain Pearson-kNN) is EPSILON-independent and reported once as
 * the reference row.
 */
public class IrrelevantNeighborInjectionEpsilonSweep {

    static String DATA_CSV = "RichMTV_clean.csv";
    static int    N_DIMS   = 8;

    static int    SHAPLEY_ITER    = 30;
    static int    SHAPLEY_SAMPLES = 200;
    static double EPSILON_1       = 0.0; // Step 2 CTXFeatSim threshold (paper: keep all co-raters)

    static double[] EPSILONS = {0.1, 0.2, 0.4, 0.6, 0.8};

    static double INJECTION_RATE = 0.5; // irrelevant duplicates injected, relative to real pool size

    static int    RUNS    = 2;
    static int    K_FOLDS = 3;
    static long[] seeds   = {1,2,3,4,5,6,7,8,9,10};

    static double SUBSAMPLE_FRACTION = 1.0; // <1.0 for a fast smoke test before the full run

    private static long   statInjectedTotal = 0;      // eps-independent (injection built before thresholding)
    private static long[] statInjectedEliminated;      // per eps
    private static double statWtotalNoCins = 0, statWinjNoCins = 0; // eps-independent
    private static double[] statWtotalCins, statWinjCins;           // per eps

    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR CINS -- Irrelevant-Neighbor Injection x EPSILON Sweep ===\n");
        int nEps = EPSILONS.length;
        statInjectedEliminated = new long[nEps];
        statWtotalCins = new double[nEps];
        statWinjCins   = new double[nEps];

        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        if (SUBSAMPLE_FRACTION < 1.0) {
            Collections.shuffle(allRows, new Random(0));
            int keep = (int) Math.round(allRows.size() * SUBSAMPLE_FRACTION);
            allRows = new ArrayList<int[]>(allRows.subList(0, keep));
            System.out.println("Subsampled to: " + allRows.size() + " rows (smoke test)");
        }

        // Layout per fold: [0,1]=MAE,RMSE noInject_NoCins; [2,3]=MAE,RMSE inject_NoCins;
        // then per eps e: [4+4e,4+4e+1]=MAE,RMSE noInject_Cins(e); [4+4e+2,4+4e+3]=MAE,RMSE inject_Cins(e)
        int width = 4 + 4 * nEps;
        double[][] allM = new double[RUNS][width];

        for (int r = 0; r < RUNS; r++) {
            int seed = (int) seeds[r % seeds.length];
            ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
            Collections.shuffle(shuffled, new Random(seed));
            int fs = shuffled.size() / K_FOLDS;

            double[] seedM = new double[width];
            for (int f = 1; f <= K_FOLDS; f++) {
                int s = (f - 1) * fs, e = (f == K_FOLDS) ? shuffled.size() : s + fs;
                ArrayList<int[]> test  = new ArrayList<int[]>(shuffled.subList(s, e));
                ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(0, s));
                if (e < shuffled.size()) train.addAll(shuffled.subList(e, shuffled.size()));

                double[] foldM = evalFold(train, test, seed, f, width, nEps);
                for (int k = 0; k < width; k++) seedM[k] += foldM[k];
            }
            for (int k = 0; k < width; k++) { seedM[k] /= K_FOLDS; allM[r][k] = seedM[k]; }
        }

        double[] avg = new double[width];
        for (int k = 0; k < width; k++) {
            double s = 0; for (int r = 0; r < RUNS; r++) s += allM[r][k];
            avg[k] = s / RUNS;
        }

        double irNoCins = (statWtotalNoCins > 0) ? statWinjNoCins / statWtotalNoCins : 0;

        System.out.println("\n=== RESULTS (avg over " + RUNS + " runs x " + K_FOLDS + " folds) ===");
        System.out.printf(Locale.US, "No-CINS (reference, EPSILON-independent): MAE noInject=%.4f, MAE +inject=%.4f (+%.4f), IR=%.2f%%%n",
            avg[0], avg[2], avg[2]-avg[0], irNoCins*100);
        System.out.println();
        System.out.printf(Locale.US, "%-8s %-12s %-12s %-10s %-10s %-10s%n",
            "EPSILON", "MAE noInj", "MAE +inj", "Elim.%", "IR(CINS)%", "IR gap(pp)");

        PrintWriter pw = new PrintWriter(new FileWriter("irrelevant_neighbor_injection_epsilon_sweep.csv"));
        pw.println("epsilon,MAE_noInject_NoCINS,MAE_inject_NoCINS,IR_NoCINS_pct,MAE_noInject_CINS,MAE_inject_CINS,elim_pct,IR_CINS_pct,IR_gap_pp");

        for (int e = 0; e < nEps; e++) {
            double maeNoInj = avg[4+4*e+0];
            double maeInj   = avg[4+4*e+2];
            double elimPct  = (statInjectedTotal > 0) ? 100.0 * statInjectedEliminated[e] / statInjectedTotal : 0;
            double irCins   = (statWtotalCins[e] > 0) ? statWinjCins[e] / statWtotalCins[e] : 0;
            double irGapPp  = (irNoCins - irCins) * 100;
            System.out.printf(Locale.US, "%-8.2f %-12.4f %-12.4f %-10.2f %-10.2f %-10.2f%n",
                EPSILONS[e], maeNoInj, maeInj, elimPct, irCins*100, irGapPp);
            pw.printf(Locale.US, "%.2f,%.6f,%.6f,%.4f,%.6f,%.6f,%.4f,%.4f,%.4f%n",
                EPSILONS[e], avg[0], avg[2], irNoCins*100, maeNoInj, maeInj, elimPct, irCins*100, irGapPp);
        }
        pw.close();
        System.out.println("\nWrote results to irrelevant_neighbor_injection_epsilon_sweep.csv");
    }

    // Returns [MAE0,RMSE0(noInject_NoCins), MAE1,RMSE1(inject_NoCins), then per eps: MAE,RMSE(noInject_Cins), MAE,RMSE(inject_Cins)]
    private static double[] evalFold(ArrayList<int[]> train, ArrayList<int[]> test, int seed, int fold, int width, int nEps) {
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

        double sumAbsNoCins0 = 0, sumSqNoCins0 = 0;
        double sumAbsNoCins1 = 0, sumSqNoCins1 = 0;
        double[] sumAbsCins0 = new double[nEps], sumSqCins0 = new double[nEps];
        double[] sumAbsCins1 = new double[nEps], sumSqCins1 = new double[nEps];
        long foldInjTotal = 0;
        long[] foldInjElim = new long[nEps];
        double foldWtotalNoCins = 0, foldWinjNoCins = 0;
        double[] foldWtotalCins = new double[nEps], foldWinjCins = new double[nEps];
        int nPred = 0;

        for (int[] tst : test) {
            int uid = tst[0];
            float uMean = uMeans.containsKey(uid) ? uMeans.get(uid) : gMean;
            int[] ctx = new int[N_DIMS];
            Arrays.fill(ctx, -1);
            for (int d = 0; d < N_DIMS; d++) if (3 + d < tst.length) ctx[d] = tst[3 + d];

            ArrayList<int[]> candidates = step2_generateNeighborhood(tst[1], ctx, g, itemIdx);
            float real = tst[2];

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

            float predNoInjNoCins, predInjNoCins;
            float[] predNoInjCins = new float[nEps];
            float[] predInjCins   = new float[nEps];

            if (validNbs.isEmpty()) {
                float fb = clamp(uMean);
                predNoInjNoCins = fb; predInjNoCins = fb;
                Arrays.fill(predNoInjCins, fb);
                Arrays.fill(predInjCins, fb);
            } else {
                predNoInjNoCins = weightedPred(uMean, gMean, uMeans, validNbs, pearsonList);

                double[] overall0 = computeOverall(fdList);
                double[] gLocal0  = computeFuzzyMeasure(fdList, pearsonList, overall0, g);
                float[] choquet0  = new float[validNbs.size()];
                for (int i = 0; i < validNbs.size(); i++) choquet0[i] = choquetIntegral(fdList.get(i), gLocal0);
                for (int e = 0; e < nEps; e++)
                    predNoInjCins[e] = weightedPredCINSEps(uMean, gMean, uMeans, validNbs, pearsonList, choquet0, EPSILONS[e]);

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
                predInjNoCins = weightedPredWithInfluence(uMean, gMean, uMeans, nbAll, pearsonAll, validNbs.size(), wStatsNoCins);
                foldWtotalNoCins += wStatsNoCins[0]; foldWinjNoCins += wStatsNoCins[1];

                double[] overallInj = computeOverall(fdAll);
                double[] gLocalInj  = computeFuzzyMeasure(fdAll, pearsonAll, overallInj, g);
                float[] choquetInj  = new float[nbAll.size()];
                for (int i = 0; i < nbAll.size(); i++) choquetInj[i] = choquetIntegral(fdAll.get(i), gLocalInj);

                for (int e = 0; e < nEps; e++) {
                    double eps = EPSILONS[e];
                    double[] wStatsCins = new double[2];
                    predInjCins[e] = weightedPredCINSWithInfluenceEps(uMean, gMean, uMeans, nbAll, pearsonAll, choquetInj, eps, validNbs.size(), wStatsCins);
                    foldWtotalCins[e] += wStatsCins[0]; foldWinjCins[e] += wStatsCins[1];

                    long elim = 0;
                    for (int i = validNbs.size(); i < nbAll.size(); i++)
                        if (choquetInj[i] < eps) elim++;
                    foldInjElim[e] += elim;
                }
                foldInjTotal += nInject;
            }

            sumAbsNoCins0 += Math.abs(predNoInjNoCins - real); sumSqNoCins0 += (predNoInjNoCins-real)*(double)(predNoInjNoCins-real);
            sumAbsNoCins1 += Math.abs(predInjNoCins - real);   sumSqNoCins1 += (predInjNoCins-real)*(double)(predInjNoCins-real);
            for (int e = 0; e < nEps; e++) {
                sumAbsCins0[e] += Math.abs(predNoInjCins[e]-real); sumSqCins0[e] += (predNoInjCins[e]-real)*(double)(predNoInjCins[e]-real);
                sumAbsCins1[e] += Math.abs(predInjCins[e]-real);   sumSqCins1[e] += (predInjCins[e]-real)*(double)(predInjCins[e]-real);
            }
            nPred++;
        }

        statInjectedTotal += foldInjTotal;
        for (int e = 0; e < nEps; e++) statInjectedEliminated[e] += foldInjElim[e];
        statWtotalNoCins += foldWtotalNoCins; statWinjNoCins += foldWinjNoCins;
        for (int e = 0; e < nEps; e++) { statWtotalCins[e] += foldWtotalCins[e]; statWinjCins[e] += foldWinjCins[e]; }

        double[] out = new double[width];
        out[0] = (nPred > 0) ? sumAbsNoCins0 / nPred : 0;
        out[1] = (nPred > 0) ? Math.sqrt(sumSqNoCins0 / nPred) : 0;
        out[2] = (nPred > 0) ? sumAbsNoCins1 / nPred : 0;
        out[3] = (nPred > 0) ? Math.sqrt(sumSqNoCins1 / nPred) : 0;
        for (int e = 0; e < nEps; e++) {
            out[4+4*e+0] = (nPred > 0) ? sumAbsCins0[e] / nPred : 0;
            out[4+4*e+1] = (nPred > 0) ? Math.sqrt(sumSqCins0[e] / nPred) : 0;
            out[4+4*e+2] = (nPred > 0) ? sumAbsCins1[e] / nPred : 0;
            out[4+4*e+3] = (nPred > 0) ? Math.sqrt(sumSqCins1[e] / nPred) : 0;
        }

        long dt = System.currentTimeMillis() - t0;
        System.out.printf(Locale.US, "SEED=%d FOLD=%d MAE(NoCINS noInj/inj)=%.4f/%.4f MAE(CINS eps0.1 noInj/inj)=%.4f/%.4f (%ds)%n",
            seed, fold, out[0], out[2], out[4], out[6], dt/1000);
        return out;
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

    /** weightedPredCINS with an explicit (swept) EPSILON instead of a fixed field. */
    private static float weightedPredCINSEps(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons, float[] choquet, double eps) {
        float sumW = 0, sumWR = 0; int numN = 0;
        for (int i = 0; i < nbs.size(); i++) {
            if (choquet[i] < eps) continue;
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

    private static float weightedPredCINSWithInfluenceEps(float uMean, float gMean, Map<Integer, Float> uMeans,
            ArrayList<int[]> nbs, ArrayList<Float> pearsons, float[] choquet, double eps, int injectStart, double[] weightStatsOut) {
        float sumW = 0, sumWR = 0; int numN = 0;
        double wTotal = 0, wInj = 0;
        for (int i = 0; i < nbs.size(); i++) {
            if (choquet[i] < eps) continue;
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
