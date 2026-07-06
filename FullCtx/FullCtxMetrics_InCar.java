import java.io.*;
import java.util.*;

/** Diagnostic only: AUC for FullCtx (r^C) on InCarMusic_clean.csv (single 80/20 split, seed=1). */
public class FullCtxMetrics_InCar {
    static int N_DIMS = 8;
    static int NEG_SAMPLE = 1000;
    static int RATING_THRESHOLD = 4;

    public static void main(String[] args) throws Exception {
        ArrayList<int[]> all = loadData("InCarMusic_clean.csv");
        System.out.println("Rows: " + all.size());

        Collections.shuffle(all, new Random(1));
        int testSize = all.size() / 5;
        ArrayList<int[]> test = new ArrayList<int[]>(all.subList(0, testSize));
        ArrayList<int[]> train = new ArrayList<int[]>(all.subList(testSize, all.size()));

        Map<Integer, ArrayList<int[]>> itemIdx = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] r : train) {
            ArrayList<int[]> l = itemIdx.get(r[1]);
            if (l == null) { l = new ArrayList<int[]>(); itemIdx.put(r[1], l); }
            l.add(r);
        }

        Map<Integer, Float> sumR = new HashMap<Integer, Float>();
        Map<Integer, Integer> cntR = new HashMap<Integer, Integer>();
        float gSum = 0; int gCnt = 0;
        for (int[] r : train) {
            sumR.merge(r[0], (float) r[2], Float::sum);
            cntR.merge(r[0], 1, Integer::sum);
            gSum += r[2]; gCnt++;
        }
        Map<Integer, Float> uMeans = new HashMap<Integer, Float>();
        for (Integer u : sumR.keySet()) uMeans.put(u, sumR.get(u) / cntR.get(u));
        float gMean = gCnt > 0 ? gSum / gCnt : 3f;

        Set<Integer> allItemSet = new HashSet<Integer>();
        for (int[] r : all) allItemSet.add(r[1]);
        ArrayList<Integer> allItems = new ArrayList<Integer>(allItemSet);

        int nTest = test.size();
        float sumAbs = 0, sumSq = 0;
        int fullMatch = 0, userMeanFallback = 0, globalMeanFallback = 0;
        for (int[] tst : test) {
            float pred = predictFullCtx(tst, itemIdx, uMeans, gMean);
            sumAbs += Math.abs(pred - tst[2]);
            sumSq += (pred - tst[2]) * (pred - tst[2]);

            ArrayList<int[]> cands = itemIdx.get(tst[1]);
            boolean found = false;
            if (cands != null) {
                for (int[] nb : cands) {
                    if (nb[0] == tst[0]) continue;
                    boolean match = true;
                    for (int d = 0; d < N_DIMS; d++) {
                        if (tst[3+d] <= 0 || nb[3+d] <= 0 || tst[3+d] != nb[3+d]) { match = false; break; }
                    }
                    if (match) { found = true; break; }
                }
            }
            if (found) { fullMatch++; continue; }
            if (uMeans.containsKey(tst[0])) userMeanFallback++; else globalMeanFallback++;
        }
        float mae = sumAbs / nTest;
        float rmse = (float) Math.sqrt(sumSq / nTest);

        Map<Integer, ArrayList<int[]>> userTest = new HashMap<Integer, ArrayList<int[]>>();
        for (int[] t : test) {
            ArrayList<int[]> l = userTest.get(t[0]);
            if (l == null) { l = new ArrayList<int[]>(); userTest.put(t[0], l); }
            l.add(t);
        }
        Map<Integer, Set<Integer>> userTrainItems = new HashMap<Integer, Set<Integer>>();
        for (int[] r : train) {
            Set<Integer> s = userTrainItems.get(r[0]);
            if (s == null) { s = new HashSet<Integer>(); userTrainItems.put(r[0], s); }
            s.add(r[1]);
        }

        Random rng = new Random(1);
        double sumAuc = 0; int nUsersAuc = 0;
        for (Map.Entry<Integer, ArrayList<int[]>> ue : userTest.entrySet()) {
            int uid = ue.getKey();
            ArrayList<int[]> testSits = ue.getValue();
            ArrayList<int[]> posSits = new ArrayList<int[]>();
            for (int[] t : testSits) if (t[2] >= RATING_THRESHOLD) posSits.add(t);
            if (posSits.isEmpty()) continue;

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
            if (negItems.isEmpty()) continue;

            int[] repSit = posSits.get(0);
            ArrayList<Float> posScores = new ArrayList<Float>();
            for (int[] sit : posSits) posScores.add(predictFullCtx(sit, itemIdx, uMeans, gMean));
            ArrayList<Float> negScores = new ArrayList<Float>();
            for (int negIid : negItems) {
                int[] negRow = repSit.clone(); negRow[1] = negIid;
                negScores.add(predictFullCtx(negRow, itemIdx, uMeans, gMean));
            }

            double wins = 0;
            for (float ps : posScores)
                for (float ns : negScores) {
                    if (ps > ns) wins += 1;
                    else if (ps == ns) wins += 0.5;
                }
            double auc = wins / ((double) posScores.size() * negScores.size());
            sumAuc += auc; nUsersAuc++;
        }
        double meanAuc = nUsersAuc > 0 ? sumAuc / nUsersAuc : 0;

        System.out.println("Test rows: " + nTest);
        System.out.printf("  FullCtx coverage (>=1 all-dims exact match): %.2f%%%n", 100.0*fullMatch/nTest);
        System.out.printf("  Fallback via UserMean (user has train history): %.2f%%%n", 100.0*userMeanFallback/nTest);
        System.out.printf("  Fallback via GlobalMean (no user history): %.2f%%%n", 100.0*globalMeanFallback/nTest);
        System.out.printf("  MAE_FullCtx  = %.6f%n", mae);
        System.out.printf("  RMSE_FullCtx = %.6f%n", rmse);
        System.out.printf("  AUC_FullCtx  = %.6f (n_users=%d)%n", meanAuc, nUsersAuc);
    }

    private static float predictFullCtx(int[] sample, Map<Integer, ArrayList<int[]>> itemIdx,
            Map<Integer, Float> uMeans, float gMean) {
        float uMean = uMeans.containsKey(sample[0]) ? uMeans.get(sample[0]) : gMean;
        ArrayList<int[]> cands = itemIdx.get(sample[1]);
        if (cands == null || cands.isEmpty()) return uMean;
        float sr = 0; int cnt = 0;
        for (int[] nb : cands) {
            if (nb[0] == sample[0]) continue;
            boolean match = true;
            for (int d = 0; d < N_DIMS; d++) {
                if (sample[3+d] <= 0 || nb[3+d] <= 0 || sample[3+d] != nb[3+d]) { match = false; break; }
            }
            if (match) { sr += nb[2]; cnt++; }
        }
        float pred = (cnt > 0) ? sr / cnt : uMean;
        return Math.max(1, Math.min(5, pred));
    }

    @SuppressWarnings("unchecked")
    private static ArrayList<int[]> loadData(String path) throws Exception {
        ArrayList<int[]> all = new ArrayList<int[]>();
        Map<String, Integer>[] encoders = new HashMap[N_DIMS];
        int[] nextCode = new int[N_DIMS];
        for (int d = 0; d < N_DIMS; d++) { encoders[d] = new HashMap<String, Integer>(); nextCode[d] = 1; }
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line; boolean first = true;
        while ((line = br.readLine()) != null) {
            line = line.trim(); if (line.isEmpty()) continue;
            if (first) { first = false; if (line.toLowerCase().contains("user")) continue; }
            String[] c;
            if (line.indexOf(';') >= 0) c = line.split(";", -1);
            else if (line.indexOf(',') >= 0) c = line.split(",", -1);
            else c = line.split("\\s+", -1);
            if (c.length < 3 + N_DIMS) continue;
            int uid = parseIntSafe(c[0].trim());
            int iid = parseIntSafe(c[1].trim());
            int rat = parseIntSafe(c[2].trim());
            if (uid < 0 || iid < 0 || rat <= 0 || rat > 5) continue;
            int[] row = new int[3 + N_DIMS];
            row[0] = uid; row[1] = iid; row[2] = rat;
            for (int d = 0; d < N_DIMS; d++) {
                String val = c[3 + d].trim();
                if (val.isEmpty() || val.equalsIgnoreCase("NA") || val.equalsIgnoreCase("null")
                        || val.equals("-1") || val.equals("\\N")) {
                    row[3 + d] = -1;
                } else {
                    int v = parseIntSafe(val);
                    if (v >= 0) {
                        row[3 + d] = v;
                    } else {
                        val = val.toLowerCase().replaceAll("\\s+", " ");
                        Integer code = encoders[d].get(val);
                        if (code == null) { code = nextCode[d]++; encoders[d].put(val, code); }
                        row[3 + d] = code;
                    }
                }
            }
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
