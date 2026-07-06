package LDOS1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import util.Pair;

public class LdosData {

    private final String csvPath;

    public Map<Integer, ArrayList<Integer>> UserIdContext;

    // situation = [0]=rating, [1]=situationCode,
    //   [2]=time, [3]=daytype, [4]=season, [5]=location, [6]=weather,
    //   [7]=social, [8]=endEmo, [9]=dominantEmo, [10]=mood,
    //   [11]=physical, [12]=decision, [13]=interaction
    // Context = indices 2..13 (12 dimensions, 49 conditions)
    public Map<Pair, ArrayList<ArrayList<Integer>>> UserMovieRatings;

    public LdosData(String csvPath) {
        this.csvPath = csvPath;
        this.UserIdContext = new HashMap<>();
        this.UserMovieRatings = new HashMap<>();
    }

    public LdosData() {
        this.csvPath = "";
        this.UserIdContext = new HashMap<>();
        this.UserMovieRatings = new HashMap<>();
    }

    public void DataLoadRating() {
        if (UserIdContext == null) UserIdContext = new HashMap<>();
        if (UserMovieRatings == null) UserMovieRatings = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            boolean first = true;

            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                if (first) {
                    first = false;
                    if (line.toLowerCase().contains("userid")) continue;
                }
                if (line.toLowerCase().startsWith("userid")) continue;

                String[] c = splitAuto(line);
                if (c.length < 19) continue; // need at least through interaction

                int userId = parseIntOrMinus1(c[0]);
                int itemId = parseIntOrMinus1(c[1]);
                int rating = parseIntOrMinus1(c[2]);
                if (userId < 0 || itemId < 0 || rating < 0) continue;

                // User demographics
                int age     = safe(c, 3);
                int sex     = safe(c, 4);
                int city    = safe(c, 5);
                int country = safe(c, 6);

                // 12 contextual dimensions (CSV columns 7-18)
                int time        = safe(c, 7);
                int daytype     = safe(c, 8);
                int season      = safe(c, 9);
                int location    = safe(c, 10);
                int weather     = safe(c, 11);
                int social      = safe(c, 12);
                int endEmo      = safe(c, 13);
                int dominantEmo = safe(c, 14);
                int mood        = safe(c, 15);
                int physical    = safe(c, 16);
                int decision    = safe(c, 17);
                int interaction = safe(c, 18);

                // User context (stable)
                if (!UserIdContext.containsKey(userId)) {
                    ArrayList<Integer> uctx = new ArrayList<>();
                    uctx.add(age); uctx.add(sex); uctx.add(city);
                    uctx.add(country); uctx.add(-1);
                    UserIdContext.put(userId, uctx);
                }

                // Situation: 14 elements [0]=rating, [1]=code, [2..13]=12 context dims
                ArrayList<Integer> one = new ArrayList<>();
                one.add(rating);      // [0]
                one.add(0);           // [1] situationCode
                one.add(time);        // [2]
                one.add(daytype);     // [3]
                one.add(season);      // [4]
                one.add(location);    // [5]
                one.add(weather);     // [6]
                one.add(social);      // [7]
                one.add(endEmo);      // [8]
                one.add(dominantEmo); // [9]
                one.add(mood);        // [10]
                one.add(physical);    // [11]
                one.add(decision);    // [12]
                one.add(interaction); // [13]

                putSituation(userId, itemId, one);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public LdosData copyExcludingPairs(Iterable<Pair> excludedPairs) {
        LdosData out = new LdosData();
        out.UserIdContext.putAll(this.UserIdContext);
        Set<Pair> excl = new HashSet<>();
        for (Pair p : excludedPairs) excl.add(p);
        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : this.UserMovieRatings.entrySet()) {
            if (!excl.contains(e.getKey()))
                out.UserMovieRatings.put(e.getKey(), e.getValue());
        }
        return out;
    }

    private void putSituation(int userId, int itemId, ArrayList<Integer> one) {
        Pair p = new Pair(userId, itemId);
        ArrayList<ArrayList<Integer>> list = UserMovieRatings.get(p);
        if (list == null) { list = new ArrayList<>(); UserMovieRatings.put(p, list); }
        list.add(one);
    }

    private int safe(String[] c, int i) {
        return (c.length > i) ? parseIntOrMinus1(c[i]) : -1;
    }

    private String[] splitAuto(String line) {
        if (line.indexOf('\t') >= 0) return line.split("\\t+", -1);
        if (line.indexOf(',') >= 0) return line.split("\\s*,\\s*", -1);
        return line.split("\\s+", -1);
    }

    private int parseIntOrMinus1(String s) {
        if (s == null) return -1;
        s = s.trim();
        if (s.isEmpty() || s.equalsIgnoreCase("NA") || s.equalsIgnoreCase("\\N")) return -1;
        try { return Integer.parseInt(s); }
        catch (Exception e) { return -1; }
    }
}