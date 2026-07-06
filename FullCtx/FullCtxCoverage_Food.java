import java.io.*;
import java.util.*;

/** Diagnostic only: measures FullCtx (r^C) coverage on Food_clean.csv. */
public class FullCtxCoverage_Food {
    static int N_DIMS = 2;

    public static void main(String[] args) throws Exception {
        ArrayList<int[]> all = loadData("Food_clean.csv");
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
        Map<String,Integer>[] encoders = new HashMap[N_DIMS];
        int[] nextCode = new int[N_DIMS];
        for (int d = 0; d < N_DIMS; d++) { encoders[d] = new HashMap<String,Integer>(); nextCode[d] = 1; }

        ArrayList<int[]> all = new ArrayList<int[]>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line; boolean first = true; String sep = ",";
        while ((line = br.readLine()) != null) {
            line = line.trim(); if (line.isEmpty()) continue;
            if (first) {
                first = false;
                if (line.contains(";")) sep = ";";
                if (line.toLowerCase().contains("user")) continue;
            }
            String[] c = line.split(sep);
            if (c.length < 3 + N_DIMS) continue;
            int uid, iid, rat;
            try { uid = Integer.parseInt(c[0].trim()); } catch(Exception ex) { continue; }
            try { iid = Integer.parseInt(c[1].trim()); } catch(Exception ex) { continue; }
            try { rat = (int) Float.parseFloat(c[2].trim()); } catch(Exception ex) { continue; }
            if (uid < 0 || iid < 0 || rat < 1 || rat > 5) continue;
            int[] row = new int[3 + N_DIMS];
            row[0] = uid; row[1] = iid; row[2] = rat;
            for (int d = 0; d < N_DIMS; d++) {
                String val = c[3+d].trim();
                if (val.isEmpty() || val.equals("-1")) { row[3+d] = -1; continue; }
                try { row[3+d] = Integer.parseInt(val); }
                catch (Exception ex) {
                    if (!encoders[d].containsKey(val)) encoders[d].put(val, nextCode[d]++);
                    row[3+d] = encoders[d].get(val);
                }
            }
            all.add(row);
        }
        br.close();
        return all;
    }
}
