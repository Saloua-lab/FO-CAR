package LDOS1;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.jdom.Document;
import org.jdom.Element;

import util.ContextNeighSim;
import util.Pair;
import util.RatingJaccard;
import util.Threshold;

public class Neighbor2 {

    public static int seuilSelection = 1;
    public static int seuil = 2;
    static int NbVoisinMax = 30;

    // dossier pour UserpBarre*.xml
    static String chemeinRepertoireUserSimilar = "D:\\datasets\\LDOS_RESULTS\\UserSimilarity\\";

    // =========================================================
    private static ArrayList<ArrayList<Integer>> getSituations(LdosData data, int userId, int movieId) {
        if (data == null || data.UserMovieRatings == null) return null;
        return data.UserMovieRatings.get(new Pair(userId, movieId));
    }

    private static int getAvgRating(ArrayList<ArrayList<Integer>> situations) {
        if (situations == null || situations.isEmpty()) return -1;
        double sum = 0; int n = 0;
        for (ArrayList<Integer> s : situations) {
            if (s != null && !s.isEmpty()) {
                int r = s.get(0);
                if (r != -1) { sum += r; n++; }
            }
        }
        if (n == 0) return -1;
        return (int) Math.round(sum / n);
    }

    // LDOS: contexte = indices 2..6 (time, weather, social, mood, location)
    private static ArrayList<Integer> extractContextFromSituation(ArrayList<Integer> s) {
        ArrayList<Integer> ctx = new ArrayList<>();
        if (s == null) return ctx;

        for (int i = 2; i <= 13; i++) {
            ctx.add(i < s.size() ? s.get(i) : -1);
        }
        return ctx;
    }

    private static ArrayList<Integer> buildTargetContext(LdosData data, int targetUser, int movieId) {
        ArrayList<ArrayList<Integer>> sits = getSituations(data, targetUser, movieId);
        if (sits == null || sits.isEmpty()) return new ArrayList<>();
        for (ArrayList<Integer> s : sits) {
            if (s != null && s.size() >= 7) return extractContextFromSituation(s);
        }
        return new ArrayList<>();
    }

    private static ArrayList<Integer> intersectionEq(ArrayList<Integer> a, ArrayList<Integer> b) {
        ArrayList<Integer> inter = new ArrayList<>();
        int n = Math.min(a.size(), b.size());
        for (int i = 0; i < n; i++) {
            int x = a.get(i), y = b.get(i);
            if (x == -1 || y == -1 || x == 0 || y == 0) inter.add(0);
            else inter.add(x == y ? 1 : 0);
        }
        return inter;
    }
    public static Map<Integer, ContextNeighSim> NeighborSelectionWithCtx(
            int targetUser, int movieId, LdosData data,
            ArrayList<Float> v, ArrayList<Integer> targetCtx) {

        Map<Integer, ContextNeighSim> listNeighbors = new HashMap<>();
        if (data == null || data.UserMovieRatings == null) return listNeighbors;
        if (targetCtx == null || targetCtx.isEmpty()) return listNeighbors;

        // Même logique que NeighborSelection mais avec targetCtx fourni
        ArrayList<Integer> nbOcc = new ArrayList<>();
        for (int i = 0; i < targetCtx.size(); i++) nbOcc.add(0);

        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e
                : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p == null || p.Id != movieId || p.UserId == targetUser) continue;
            for (ArrayList<Integer> s : e.getValue()) {
                ArrayList<Integer> ctx = extractContextFromSituation(s);
                int n = Math.min(ctx.size(), targetCtx.size());
                for (int i = 0; i < n; i++) {
                    int a = ctx.get(i), b = targetCtx.get(i);
                    if (a != -1 && b != -1 && a != 0 && b != 0)
                        nbOcc.set(i, nbOcc.get(i) + 1);
                }
            }
        }

        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e
                : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p == null || p.Id != movieId || p.UserId == targetUser) continue;

            ArrayList<ArrayList<Integer>> sits = e.getValue();
            if (sits == null || sits.isEmpty()) continue;

            float bestSim = 0f;
            ArrayList<Integer> bestCtx = null;
            for (ArrayList<Integer> s : sits) {
                ArrayList<Integer> vctx = extractContextFromSituation(s);
                float sim = computeSimilarity(targetCtx, vctx, v, nbOcc);
                if (sim > bestSim) { bestSim = sim; bestCtx = vctx; }
            }

            int rVoisin = getAvgRating(sits);
            if (rVoisin == -1) continue;

            if (bestSim >= 0.0f) {
                listNeighbors.put(p.UserId,
                    new ContextNeighSim(bestCtx, bestSim, rVoisin));
            }
        }
        return listNeighbors;
    }

    private static float computeSimilarity(ArrayList<Integer> ctxTarget,
                                           ArrayList<Integer> ctxOther,
                                           ArrayList<Float> v,
                                           ArrayList<Integer> nbOccContextAttribut) {
        if (ctxTarget == null || ctxOther == null || v == null) return 0f;

        ArrayList<Integer> inter = intersectionEq(ctxTarget, ctxOther);

        float num = 0f, den = 0f;
        int n = Math.min(inter.size(), v.size());

        for (int i = 0; i < n; i++) {
            if (nbOccContextAttribut != null && i < nbOccContextAttribut.size()) {
                if (nbOccContextAttribut.get(i) < seuilSelection) continue;
            }
            den += v.get(i);
            if (inter.get(i) == 1) num += v.get(i);
        }
        return (den == 0f) ? 0f : (num / den);
    }

    private static float computeSimilarity2(ArrayList<Integer> ctxTarget,
                                            ArrayList<Integer> ctxOther,
                                            ArrayList<Integer> nbOccContextAttribut,
                                            ArrayList<Float> v) {
        if (ctxTarget == null || ctxOther == null || v == null) return 0f;

        float inter = 0f, union = 0f;
        int n = Math.min(Math.min(ctxTarget.size(), ctxOther.size()), v.size());

        for (int i = 0; i < n; i++) {
            if (nbOccContextAttribut != null && i < nbOccContextAttribut.size()) {
                if (nbOccContextAttribut.get(i) <= seuil) continue;
            }

            int a = ctxTarget.get(i);
            int b = ctxOther.get(i);

            if (a == -1 || b == -1 || a == 0 || b == 0) continue;

            union += v.get(i);
            if (a == b) inter += v.get(i);
        }
        return (union == 0f) ? 0f : (inter / union);
    }

    // =========================================================
    public static Map<Integer, ContextNeighSim> NeighborSelection(int targetUser,
                                                                  int movieId,
                                                                  LdosData data,
                                                                  ArrayList<Float> v)
            throws NumberFormatException, IOException {

        Map<Integer, ContextNeighSim> listNeighbors = new HashMap<>();
        if (data == null || data.UserMovieRatings == null) return listNeighbors;

        ArrayList<Integer> targetCtx = buildTargetContext(data, targetUser, movieId);
        if (targetCtx == null || targetCtx.isEmpty()) return listNeighbors;

        ArrayList<Integer> nbOcc = new ArrayList<>();
        for (int i = 0; i < targetCtx.size(); i++) nbOcc.add(0);

        // compter occurrences comparables
        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p == null) continue;
            if (p.Id != movieId) continue;
            if (p.UserId == targetUser) continue;

            ArrayList<ArrayList<Integer>> sits = e.getValue();
            if (sits == null) continue;

            for (ArrayList<Integer> s : sits) {
                ArrayList<Integer> ctx = extractContextFromSituation(s);
                int n = Math.min(ctx.size(), targetCtx.size());
                for (int i = 0; i < n; i++) {
                    int a = ctx.get(i), b = targetCtx.get(i);
                    if (a != -1 && b != -1 && a != 0 && b != 0) nbOcc.set(i, nbOcc.get(i) + 1);
                }
            }
        }

        // voisins
        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p == null) continue;
            if (p.Id != movieId) continue;

            int voisinId = p.UserId;
            if (voisinId == targetUser) continue;

            ArrayList<ArrayList<Integer>> sits = e.getValue();
            if (sits == null || sits.isEmpty()) continue;

            float bestSim = 0f;
         // Juste après "float bestSim = 0f;"
            int debugCount = 0;
            ArrayList<Integer> bestCtx = null;

            for (ArrayList<Integer> s : sits) {
                ArrayList<Integer> vctx = extractContextFromSituation(s);
                float sim = computeSimilarity(targetCtx, vctx, v, nbOcc);
                if (sim > bestSim) { bestSim = sim; bestCtx = vctx; }
            }

            int rVoisin = getAvgRating(sits);
            if (rVoisin == -1) continue;
         // Juste avant "if (bestSim >= Threshold.epsilon1)"
            if (debugCount < 3) {
                System.out.println("  NEIGH " + voisinId + " bestSim=" + bestSim + " rVoisin=" + rVoisin);
                debugCount++;
            }
            if (bestSim >= Threshold.epsilon1) {
                listNeighbors.put(voisinId, new ContextNeighSim(bestCtx, bestSim, rVoisin));
            }
        }
        System.out.println("DEBUG NS: targetUser=" + targetUser 
        	    + " movieId=" + movieId 
        	    + " targetCtx=" + targetCtx 
        	    + " totalPairsChecked=" + data.UserMovieRatings.size()
        	    + " neighborsFound=" + listNeighbors.size());
        return listNeighbors;
    }

    // =========================================================
    static void generateFilePpBarre(LdosData data,
                                    Map<Integer, ContextNeighSim> ListeVoisins,
                                    int targetUser,
                                    ArrayList<Float> v,
                                    int compteur) throws IOException {

        Document doc = new Document();
        Element root = new Element("users");
        doc.setRootElement(root);

        Iterator<Map.Entry<Integer, ContextNeighSim>> it = ListeVoisins.entrySet().iterator();
        int nb = 0;

        while (it.hasNext() && nb < NbVoisinMax) {
            Map.Entry<Integer, ContextNeighSim> entry = it.next();
            int voisinId = entry.getKey();

            ArrayList<RatingJaccard> ratingJaccard = getListeRatingJaccard(targetUser, voisinId, data, v);

            float pBarre = 0f;
            for (RatingJaccard rj : ratingJaccard) pBarre += (rj.rating * rj.SimJaccard);
            if (!ratingJaccard.isEmpty()) pBarre = pBarre / ratingJaccard.size();

            Element userNouv = new Element("user");
            userNouv.setAttribute("user_id", String.valueOf(voisinId));
            userNouv.setAttribute("pBarre", String.valueOf(pBarre));
            root.addContent(userNouv);

            nb++;
        }

        try {
            User2.enregistreFichier(chemeinRepertoireUserSimilar + "UserpBarre" + compteur + ".xml", doc);
        } catch (Exception ex) {
            // ignore
        }
    }

    // =========================================================
    static ArrayList<RatingJaccard> getListeRatingJaccard(int targetUser,
                                                          int voisinId,
                                                          LdosData data,
                                                          ArrayList<Float> v) {

        ArrayList<RatingJaccard> out = new ArrayList<>();
        if (data == null || data.UserMovieRatings == null) return out;

        Map<Integer, ArrayList<ArrayList<Integer>>> targetItems = new HashMap<>();
        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p != null && p.UserId == targetUser) targetItems.put(p.Id, e.getValue());
        }

        for (Map.Entry<Pair, ArrayList<ArrayList<Integer>>> e : data.UserMovieRatings.entrySet()) {
            Pair p = e.getKey();
            if (p == null) continue;
            if (p.UserId != voisinId) continue;

            int itemId = p.Id;
            ArrayList<ArrayList<Integer>> voisinSits = e.getValue();
            ArrayList<ArrayList<Integer>> targetSits = targetItems.get(itemId);

            if (targetSits == null || targetSits.isEmpty() || voisinSits == null || voisinSits.isEmpty()) continue;

            ArrayList<Integer> targetCtx = extractContextFromSituation(targetSits.get(0));
            if (targetCtx == null || targetCtx.isEmpty()) continue;

            ArrayList<Integer> nbOcc = new ArrayList<>();
            for (int i = 0; i < targetCtx.size(); i++) nbOcc.add(0);

            for (ArrayList<Integer> s : voisinSits) {
                ArrayList<Integer> vctx = extractContextFromSituation(s);
                int n = Math.min(vctx.size(), targetCtx.size());
                for (int i = 0; i < n; i++) {
                    int a = vctx.get(i), b = targetCtx.get(i);
                    if (a != -1 && b != -1 && a != 0 && b != 0) nbOcc.set(i, nbOcc.get(i) + 1);
                }
            }

            float bestSim = 0f;
            for (ArrayList<Integer> s : voisinSits) {
                ArrayList<Integer> vctx = extractContextFromSituation(s);
                float sim = computeSimilarity2(targetCtx, vctx, nbOcc, v);
                if (sim > bestSim) bestSim = sim;
            }

            if (bestSim > Threshold.epsilon2) {
                int rVoisin = getAvgRating(voisinSits);
                if (rVoisin != -1) out.add(new RatingJaccard(rVoisin, bestSim));
            }
        }

        return out;
    }

    // =========================================================
    public static ArrayList<Integer> getContextByUser(LdosData data, int userId) {
        if (data == null || data.UserIdContext == null) return new ArrayList<>();

        Object raw = data.UserIdContext.get(userId);
        if (raw == null) return new ArrayList<>();

        ArrayList<Integer> out = new ArrayList<>();

        if (raw instanceof List<?>) {
            for (Object o : (List<?>) raw) {
                out.add((o instanceof Integer) ? (Integer) o : -1);
            }
            // padding si besoin (computeSimilarity utilise 5 dims)
            while (out.size() < 5) out.add(-1);
            if (out.size() > 5) out = new ArrayList<>(out.subList(0, 5));
            return out;
        }

        return new ArrayList<>();
    }

}
