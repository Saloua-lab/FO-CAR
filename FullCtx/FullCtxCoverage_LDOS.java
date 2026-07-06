import java.io.*;
import java.util.*;

/** Diagnostic only: measures FullCtx (r^C) coverage on LDOS-CoMoDa.csv. */
public class FullCtxCoverage_LDOS {
    static int N_DIMS = 5;
    static final int[] CTX_COLS = {7, 10, 11, 12, 15};  // time, location, weather, social, mood

    public static void main(String[] args) throws Exception {
        ArrayList<int[]> all = loadData("LDOS-CoMoDa.csv");
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

        int nTest = test.size();
        int noCoraters = 0, coratersButNoMatch = 0, fullMatch = 0;

        for (int[] tst : test) {
            ArrayList<int[]> cands = itemIdx.get(tst[1]);
            if (cands == null || cands.isEmpty()) { noCoraters++; continue; }
            boolean found = false;
            for (int[] nb : cands) {
                if (nb[0] == tst[0]) continue;
                boolean match = true;
                for (int d = 0; d < N_DIMS; d++) {
                    if (tst[3+d] <= 0 || nb[3+d] <= 0 || tst[3+d] != nb[3+d]) { match = false; break; }
                }
                if (match) { found = true; break; }
            }
            if (found) fullMatch++; else coratersButNoMatch++;
        }

        System.out.println("Test rows: " + nTest);
        System.out.printf("  FullCtx coverage (>=1 all-dims exact match): %d (%.2f%%)%n", fullMatch, 100.0*fullMatch/nTest);
        System.out.printf("  Has co-raters but NO all-dims match (uMean fallback): %d (%.2f%%)%n", coratersButNoMatch, 100.0*coratersButNoMatch/nTest);
        System.out.printf("  No co-raters at all (uMean fallback): %d (%.2f%%)%n", noCoraters, 100.0*noCoraters/nTest);
        System.out.printf("  Total uMean fallback rate: %.2f%%%n", 100.0*(coratersButNoMatch+noCoraters)/nTest);
    }

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
            if (uid < 0 || iid < 0 || rat < 0) continue;
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
