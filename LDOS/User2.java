package LDOS1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import util.Pair;
import util.Parsing;
import util.Threshold;

public class User2 {

    static String chemeinRepertoireUserSimilar =
            "D:/ma thÃ¨se/ma thÃ¨se/recommendation/Experimental study/UserSimilarity/";

    // =========================================================
    // Helpers (LDOS)
    private static ArrayList<ArrayList<Integer>> getSituations(LdosData data, int userId, int movieId) {
        Pair p = new Pair(userId, movieId);
        return data.UserMovieRatings.get(p);
    }

    private static int getAvgRating(ArrayList<ArrayList<Integer>> situations) {
        if (situations == null || situations.isEmpty()) return -1;
        double sum = 0;
        int n = 0;
        for (ArrayList<Integer> s : situations) {
            if (s != null && !s.isEmpty()) {
                int r = s.get(0); // rating index 0
                if (r != -1) {
                    sum += r;
                    n++;
                }
            }
        }
        if (n == 0) return -1;
        return (int) Math.round(sum / n);
    }

    // âœ… LDOS: extrait contexte situationnel (5 dims) depuis une situation
    // situation = [0]=rating, [1]=dummy, [2..6]=5 dims
    private static ArrayList<Integer> extractContext(ArrayList<Integer> s) {
        ArrayList<Integer> ctx = new ArrayList<>();
        if (s == null) return ctx;
        for (int k = 2; k <= 6 && k < s.size(); k++) ctx.add(s.get(k));
        return ctx;
    }

    // âœ… prend la 1Ã¨re situation exploitable pour construire un contexte (5 dims)
    private static ArrayList<Integer> getFirstContextForUserItem(LdosData data, int userId, int itemId) {
        ArrayList<ArrayList<Integer>> sits = getSituations(data, userId, itemId);
        if (sits == null || sits.isEmpty()) return new ArrayList<>();

        for (ArrayList<Integer> s : sits) {
            if (s != null && s.size() >= 7) {
                return extractContext(s);
            }
        }
        return new ArrayList<>();
    }

    // =========================================================
    public static void enregistreFichierWithCondition(String fichier, Document doc) {
        try {
            XMLOutputter sortie = new XMLOutputter(Format.getPrettyFormat());
            File monFichier = new File(fichier);
            if (monFichier.exists()) {
                Document doc1 = Parsing.getDocument(monFichier);
                Element root = doc1.getRootElement();
                Element rootNouv = doc.getRootElement();
                Element user = rootNouv.getChild("user");
                Element userCopie = (Element) user.clone();
                root.addContent(userCopie);
                sortie.output(doc1, new FileOutputStream(fichier));
            } else {
                sortie.output(doc, new FileOutputStream(fichier));
            }
        } catch (Exception exp) {
        }
    }

    public static void enregistreFichier(String fichier, Document doc) {
        try {
            XMLOutputter sortie = new XMLOutputter(Format.getPrettyFormat());
            sortie.output(doc, new FileOutputStream(fichier));
        } catch (Exception exp) {
        }
    }

    // =========================================================
    // SimilaritÃ© contexte pour baseline (targetContext vs contextCourant)
    public static float simCxtuserCxtUserdifferentMovies(ArrayList<Integer> targetContext,
                                                         ArrayList<Integer> contextCourant,
                                                         ArrayList<Float> v) {

        ArrayList<Integer> inter = new ArrayList<Integer>();
        int n = Math.min(targetContext.size(), contextCourant.size());

        // âš ï¸� ton ancien code mettait 1 si "non nul" (pas Ã©galitÃ©)
        // je garde identique pour ne pas changer tes rÃ©sultats
        for (int i = 0; i < n; i++) {
            if (targetContext.get(i) != 0 && contextCourant.get(i) != 0) inter.add(1);
            else inter.add(0);
        }

        float som = 0;
        for (int i = 0; i < inter.size() && i < v.size(); i++) {
            float val = (float) inter.get(i);
            som = som + (val * v.get(i));
        }

        float union = 0;
        for (int i = 0; i < v.size(); i++) union = union + v.get(i);

        if (union == 0) return 0;
        return som / union;
    }

    // =========================================================
    // BASELINE USER (LDOS)
    public static float getUserBaseline(int idTargetUser, ArrayList<Integer> targetContext,
                                        int movieId, LdosData data, ArrayList<Float> v) {

        float som = 0;
        int cnt = 0;

        Iterator<Map.Entry<Pair, ArrayList<ArrayList<Integer>>>> iterator =
                data.UserMovieRatings.entrySet().iterator();

        while (iterator.hasNext()) {
            Map.Entry<Pair, ArrayList<ArrayList<Integer>>> entry = iterator.next();
            Pair key = entry.getKey();

            if (key.UserId == idTargetUser) {

                ArrayList<ArrayList<Integer>> situations = entry.getValue();
                if (situations == null || situations.isEmpty()) continue;

                for (ArrayList<Integer> s : situations) {
                    if (s == null || s.size() < 7) continue;

                    ArrayList<Integer> contextCourant = extractContext(s);
                    float sim = simCxtuserCxtUserdifferentMovies(targetContext, contextCourant, v);

                    if (sim >= Threshold.epsilon3) {
                        som += s.get(0); // rating
                        cnt++;
                    }
                }
            }
        }

        if (cnt == 0) return 0;
        return som / cnt;
    }

    // =========================================================
    static void repertoireRatingBarreUser() throws IOException {

        float som = 0, xbarre = 0;
        int compteur = 0;
        Document doc1 = new Document();
        Element root = new Element("users");
        doc1.setRootElement(root);

        // âš ï¸� LDOS-CoMoDa : 121 users (d'aprÃ¨s ton log CARSKit)
        for (int i = 1; i <= 121; i++) {
            File monFichier = new File(chemeinRepertoireUserSimilar + "user" + i + ".xml");
            if (monFichier.exists()) {

                som = 0;
                xbarre = 0;
                compteur = 0;

                Document doc = Parsing.getDocument(monFichier);
                Element root1 = doc.getRootElement();
                List listuser = root1.getChildren("user");
                Iterator it3 = listuser.iterator();
                String userId = null;

                while (it3.hasNext()) {
                    Element elem = (Element) it3.next();
                    Element rating = elem.getChild("rating");
                    userId = elem.getAttributeValue("user_id");
                    String val = rating.getAttributeValue("rating");
                    som += Integer.parseInt(val);
                    compteur++;
                }

                if (compteur != 0) xbarre = som / compteur;
                else xbarre = 0;

                Element userNouv = new Element("user");
                userNouv.setAttribute("user_id", userId);
                userNouv.setAttribute("xbarre", String.valueOf(xbarre));
                root.addContent(userNouv);
            }
        }

        try {
            enregistreFichier(chemeinRepertoireUserSimilar + "userXbarre.xml", doc1);
        } catch (Exception ex) {
        }
    }

    static float computeSimilarity(ArrayList<Integer> ctxTargetUser, ArrayList<Integer> ctxUser, ArrayList<Float> v) {
        float SimCtxt = 0;
        ArrayList<Integer> ctxIntersection = chercherIntersection(ctxTargetUser, ctxUser);
        float som = 0;

        for (int i = 0; i < 5 && i < ctxIntersection.size() && i < v.size(); i++) {
            float val = (float) ctxIntersection.get(i);
            som = som + (val * v.get(i));
        }

        float union = 0;
        for (int i = 0; i < v.size(); i++) union = union + v.get(i);

        if (union == 0) return 0;
        SimCtxt = som / union;
        return SimCtxt;
    }

    static ArrayList<Integer> chercherIntersection(ArrayList<Integer> ctxTargetUser, ArrayList<Integer> ctxUser) {
        ArrayList<Integer> inter = new ArrayList<Integer>();
        int n = Math.min(ctxTargetUser.size(), ctxUser.size());
        for (int i = 0; i < n; i++) {
            if (ctxTargetUser.get(i) != 0 && ctxUser.get(i) != 0) inter.add(1);
            else inter.add(0);
        }
        return inter;
    }

    // =========================================================
    // repertoire() (LDOS)
    // âš ï¸� Ici ton ancien code utilisait Neighbor.getContextByUser() (UserIdContext),
    // mais LDOS n'a pas Ã§a. Donc on construit un contexte situationnel 5 dims.
    static void repertoire(LdosData data, ArrayList<Float> v) throws IOException {

        String s = "";
        FileWriter fw = new FileWriter(chemeinRepertoireUserSimilar + "rataing.txt", true);
        BufferedWriter output = new BufferedWriter(fw);

        Iterator<Map.Entry<Pair, ArrayList<ArrayList<Integer>>>> iterator =
                data.UserMovieRatings.entrySet().iterator();

        while (iterator.hasNext()) {
            Map.Entry<Pair, ArrayList<ArrayList<Integer>>> entry = iterator.next();
            Pair key = entry.getKey();
            int targetUser = key.UserId;
            int targetItem = key.Id;

            int rati = getAvgRating(entry.getValue());

            s = "targetUser: " + targetUser + " targetItem: " + targetItem + "\n";
            output.write(s);
            output.flush();

            org.jdom.Document newDocUser = new Document();
            Element rootUser = new Element("UserSim");
            newDocUser.setRootElement(rootUser);

            Element user = new Element("user");
            user.setAttribute("user_id", String.valueOf(targetUser));
            user.setAttribute("movie_id", String.valueOf(targetItem));
            rootUser.addContent(user);

            Element rating = new Element("rating");
            user.addContent(rating);
            rating.setAttribute("rating", String.valueOf(rati));

            // contexte situationnel du targetUser pour cet item
            ArrayList<Integer> ctxT1 = getFirstContextForUserItem(data, targetUser, targetItem);

            // comparer avec autres users qui ont le mÃªme item
            Iterator<Map.Entry<Pair, ArrayList<ArrayList<Integer>>>> iterator1 =
                    data.UserMovieRatings.entrySet().iterator();

            while (iterator1.hasNext()) {
                Map.Entry<Pair, ArrayList<ArrayList<Integer>>> entry1 = iterator1.next();
                Pair key1 = entry1.getKey();

                int targetUser1 = key1.UserId;
                int targetItem1 = key1.Id;

                if (targetUser1 != targetUser && targetItem1 == targetItem) {

                    int rati1 = getAvgRating(entry1.getValue());

                    ArrayList<Integer> ctxT2 = getFirstContextForUserItem(data, targetUser1, targetItem1);

                    float sim = computeSimilarity(ctxT1, ctxT2, v);

                    if (sim >= Threshold.epsilon4) {
                        Element userSimilar = new Element("user");
                        userSimilar.setAttribute("user_id", String.valueOf(targetUser1));
                        userSimilar.setAttribute("rating", String.valueOf(rati1));
                        userSimilar.setAttribute("sim", String.valueOf(sim));
                        user.addContent(userSimilar);
                    }
                }
            }

            try {
                enregistreFichierWithCondition(chemeinRepertoireUserSimilar + "user" + targetUser + ".xml", newDocUser);
            } catch (Exception ex) {
            }
        }

        output.close();
    }

    static float getXBarre(int id) {
        float xb = 0;
        File monFichier = new File(chemeinRepertoireUserSimilar + "userXbarre.xml");
        if (monFichier.exists()) {
            Document doc1 = Parsing.getDocument(monFichier);
            Element root = doc1.getRootElement();
            List ListUsers = root.getChildren();
            Iterator it = ListUsers.iterator();

            while (it.hasNext()) {
                Element elem = (Element) it.next();
                if (Integer.valueOf(elem.getAttributeValue("user_id")) == id) {
                    xb = Float.valueOf(elem.getAttributeValue("xbarre"));
                    return xb;
                }
            }
        }
        return xb;
    }

    public static float UserSimilarityPerason(int idTargetUser, int idVoisin, ArrayList<Float> v) {
        float pearsonSim = 0;
        Map<Pair, ArrayList<Float>> targetUser1MovieUser2RatingJaccSim = new HashMap<Pair, ArrayList<Float>>();

        File monFichier = new File(chemeinRepertoireUserSimilar + "user" + idTargetUser + ".xml");
        if (monFichier.exists()) {

            Document doc1 = Parsing.getDocument(monFichier);
            Element root = doc1.getRootElement();
            List mUser = root.getChildren();
            Iterator it = mUser.iterator();

            while (it.hasNext()) {
                Element elem = (Element) it.next();
                List vUser = elem.getChildren("user");

                int ratingTargetUser = Integer.valueOf((elem.getChild("rating")).getAttributeValue("rating"));
                Iterator it1 = vUser.iterator();

                while (it1.hasNext()) {
                    Element elem1 = (Element) it1.next();
                    if (Integer.valueOf(elem1.getAttributeValue("user_id")) == idVoisin) {

                        int ratingVoisinUser = Integer.valueOf(elem1.getAttributeValue("rating"));
                        float SimJacc = Float.valueOf(elem1.getAttributeValue("sim"));

                        Pair p = new Pair(idTargetUser, ratingTargetUser);
                        ArrayList<Float> array = new ArrayList<Float>();

                        Float idU = Float.valueOf(elem1.getAttributeValue("user_id"));
                        array.add(0, idU);
                        array.add(1, (float) ratingTargetUser);
                        array.add(2, (float) ratingVoisinUser);
                        array.add(3, SimJacc);

                        targetUser1MovieUser2RatingJaccSim.put(p, array);
                    }
                }
            }
        }

        float num = 0;
        Iterator iterator = targetUser1MovieUser2RatingJaccSim.entrySet().iterator();

        float targetUserbarre = getXBarre(idTargetUser);
        float targetVoisinbarre = getXBarre(idVoisin);

        float part1 = 0, part2 = 0, part3 = 0;

        while (iterator.hasNext()) {
            Map.Entry entry = (Map.Entry) iterator.next();
            ArrayList<Float> value = (ArrayList<Float>) entry.getValue();

            float ratingTargetUser = value.get(1);
            float ratingVoisinUser = value.get(2);
            float SimJacc = value.get(3);

            float a = ratingTargetUser - targetUserbarre;
            float b = ratingVoisinUser - targetVoisinbarre;

            num = num + a * b * SimJacc;
            part1 = part1 + (a * a);
            part2 = part2 + (b * b);
            part3 = part3 + (SimJacc * SimJacc);
        }

        if ((part1 * part2 * part3) != 0) {
            pearsonSim = num / (part1 * part2 * part3);
        }

        return pearsonSim;
    }

    public static void main(String[] args) throws IOException {
        LdosData data = new LdosData();
        //data.DataLoadUser();
        //data.DataLoadMovie();
        data.DataLoadRating();

        // repertoire(data, v) : se lance depuis ton pipeline si besoin
        // repertoireRatingBarreUser();
    }
}
