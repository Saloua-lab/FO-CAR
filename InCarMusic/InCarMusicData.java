package InCarMusic1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import util.Pair;

/**
 * Data loader for InCarMusic dataset.
 * 
 * CSV: userID, itemID, rating, DrivingStyle, landscape, mood,
 *      naturalphenomena, RoadType, sleepiness, trafficConditions, weather
 * 
 * Situation array: [0]=rating, [1]=code, [2..9] = 8 context dimensions
 * Context dimensions (8, 26 conditions):
 *   [2] DrivingStyle     (1=relaxed, 2=sport)
 *   [3] landscape        (1=coast, 2=country, 3=mountains, 4=urban)
 *   [4] mood             (1=active, 2=happy, 3=lazy, 4=sad)
 *   [5] naturalphenomena (1=morning, 2=daytime, 3=afternoon, 4=night)
 *   [6] RoadType         (1=city, 2=highway, 3=serpentine)
 *   [7] sleepiness       (1=awake, 2=sleepy)
 *   [8] trafficConditions(1=free, 2=lots, 3=jam)
 *   [9] weather          (1=cloudy, 2=rainy, 3=snowing, 4=sunny)
 */
public class InCarMusicData {

    private final String csvPath;
    public Map<Integer, ArrayList<Integer>> UserIdContext;
    public Map<Pair, ArrayList<ArrayList<Integer>>> UserMovieRatings;

    public static final int N_CTX_DIMS = 8;

    public InCarMusicData(String csvPath) {
        this.csvPath = csvPath;
        this.UserIdContext = new HashMap<>();
        this.UserMovieRatings = new HashMap<>();
    }

    public InCarMusicData() {
        this.csvPath = "";
        this.UserIdContext = new HashMap<>();
        this.UserMovieRatings = new HashMap<>();
    }

    public void DataLoadRating() {
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

                String[] c = splitAuto(line);
                if (c.length < 11) continue; // need userID, itemID, rating + 8 ctx

                int userId = parseIntOrMinus1(c[0]);
                int itemId = parseIntOrMinus1(c[1]);
                int rating = parseIntOrMinus1(c[2]);
                if (userId < 0 || itemId < 0 || rating < 0) continue;

                // 8 context dimensions (CSV columns 3-10)
                int drivingStyle     = safe(c, 3);
                int landscape        = safe(c, 4);
                int mood             = safe(c, 5);
                int naturalPhenomena = safe(c, 6);
                int roadType         = safe(c, 7);
                int sleepiness       = safe(c, 8);
                int trafficCond      = safe(c, 9);
                int weather          = safe(c, 10);

                // User context (placeholder)
                if (!UserIdContext.containsKey(userId)) {
                    ArrayList<Integer> uctx = new ArrayList<>();
                    for (int i = 0; i < 5; i++) uctx.add(-1);
                    UserIdContext.put(userId, uctx);
                }

                // Situation: [0]=rating, [1]=code, [2..9]=8 context dims
                ArrayList<Integer> one = new ArrayList<>();
                one.add(rating);          // [0]
                one.add(0);               // [1] situationCode
                one.add(drivingStyle);    // [2]
                one.add(landscape);       // [3]
                one.add(mood);            // [4]
                one.add(naturalPhenomena);// [5]
                one.add(roadType);        // [6]
                one.add(sleepiness);      // [7]
                one.add(trafficCond);     // [8]
                one.add(weather);         // [9]

                putSituation(userId, itemId, one);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public InCarMusicData copyExcludingPairs(Iterable<Pair> excludedPairs) {
        InCarMusicData out = new InCarMusicData();
        out.UserIdContext.putAll(this.UserIdContext);
        Set<Pair> excl = new HashSet<>();
        for (Pair p : excludedPairs) excl.add(p);
        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : this.UserMovieRatings.entrySet())
            if (!excl.contains(e.getKey()))
                out.UserMovieRatings.put(e.getKey(), e.getValue());
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
