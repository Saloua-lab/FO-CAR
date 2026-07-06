package Food1;

import java.io.*;
import java.util.*;

/**
 * FO-CAR Pipeline for Food (AIST) dataset
 * 2 context dims (situation, virtualflag)
 * 6360 ratings, 212 users, 20 items, 100% context
 * Pipeline: Shapley -> Choquet+Sugeno -> sim^3 -> mean-centered
 * 
 */
public class RecomMain2_FOCAR_Food {

    public static String DATA_CSV = "D:\\datasets\\Food_clean.csv";
    public static int    K_FOLDS  = 3;
    public static int    NB_RUNS  = 10;
    public static long[] seeds    = {1,2,3,4,5,6,7,8,9,10};
    public static int    N_DIMS   = 2;

    private static float lastMAE, lastRMSE;
    private static double[] lastMetrics;

    public static void main(String[] args) throws Exception {
        System.out.println("=== FO-CAR Food (2 dims) ===\n");
        ArrayList<int[]> allRows = loadData(DATA_CSV);
        System.out.println("Total rows: " + allRows.size());

        double[][] allM = new double[NB_RUNS][8];
        for (int r = 0; r < NB_RUNS; r++) {
            int seed = (int)seeds[r];
            ArrayList<int[]> shuffled = new ArrayList<int[]>(allRows);
            Collections.shuffle(shuffled, new Random(seed));
            int fs = shuffled.size() / K_FOLDS;
            double[] seedM = new double[8];
            for (int f=1; f<=K_FOLDS; f++) {
                int s=(f-1)*fs, e=(f==K_FOLDS)?shuffled.size():s+fs;
                ArrayList<int[]> test = new ArrayList<int[]>(shuffled.subList(s,e));
                ArrayList<int[]> train = new ArrayList<int[]>(shuffled.subList(0,s));
                if (e<shuffled.size()) train.addAll(shuffled.subList(e,shuffled.size()));
                evalFold(train, test, seed, f);
                for (int m=0;m<8;m++) seedM[m]+=lastMetrics[m];
            }
            for (int m=0;m<8;m++){seedM[m]/=K_FOLDS;allM[r][m]=seedM[m];}
            System.out.printf("SEED=%d MAE=%.4f RMSE=%.4f P@5=%.4f R@5=%.4f N@5=%.4f%n%n",
                seed,seedM[0],seedM[1],seedM[2],seedM[4],seedM[6]);
        }
        System.out.println("\n=== RESULTS ===");
        String[] names={"MAE","RMSE","Prec@5","Prec@10","Rec@5","Rec@10","NDCG@5","NDCG@10"};
        for(int m=0;m<8;m++){
            double[] col=new double[NB_RUNS];
            for(int r=0;r<NB_RUNS;r++) col[r]=allM[r][m];
            System.out.printf("  %-10s = %.6f +/- %.6f%n",names[m],mean2(col),std2(col));
        }
        System.out.println("\nTarget: MAE=0.901 RMSE=1.051 P@5=0.092 P@10=0.095 R@5=0.198 R@10=0.205 N@5=0.302 N@10=0.152");
    }

    static double mean2(double[] v){double s=0;for(double x:v)s+=x;return s/v.length;}
    static double std2(double[] v){double m=mean2(v),s=0;for(double x:v)s+=(x-m)*(x-m);return Math.sqrt(s/v.length);}

    private static ArrayList<int[]> loadData(String path) throws Exception {
        ArrayList<int[]> all = new ArrayList<int[]>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line; boolean first=true;
        while ((line=br.readLine())!=null) {
            line=line.trim(); if(line.isEmpty()) continue;
            if(first){first=false; if(line.toLowerCase().contains("user")) continue;}
            String[] c=line.split(",");
            if(c.length<3+N_DIMS) continue;
            int[] row=new int[3+N_DIMS];
            for(int i=0;i<row.length;i++){
                try{row[i]=(int)Float.parseFloat(c[i].trim());}catch(Exception ex){row[i]=-1;}
            }
            if(row[0]<0||row[1]<0||row[2]<0) continue;
            all.add(row);
        }
        br.close();
        return all;
    }

    private static void evalFold(ArrayList<int[]> train, ArrayList<int[]> test,
            int seed, int fold) {
		// Item index
		Map<Integer,ArrayList<int[]>> itemIdx = new HashMap<Integer,ArrayList<int[]>>();
		for(int[] r:train){
		ArrayList<int[]> l=itemIdx.get(r[1]);
		if(l==null){l=new ArrayList<int[]>();itemIdx.put(r[1],l);}
		l.add(r);
		}
		
		// User means
		Map<Integer,float[]> uAcc = new HashMap<Integer,float[]>();
		for(int[] r:train){
		float[] a=uAcc.get(r[0]);
		if(a==null){a=new float[]{0,0};uAcc.put(r[0],a);}
		a[0]+=r[2]; a[1]++;
		}
		Map<Integer,Float> uMeans = new HashMap<Integer,Float>();
		float gSum=0; int gN=0;
		for(Map.Entry<Integer,float[]> e:uAcc.entrySet()){
		float m=e.getValue()[0]/e.getValue()[1];
		uMeans.put(e.getKey(),m); gSum+=m; gN++;
		}
		float gMean=(gN>0)?gSum/gN:3f;
		
		// Training items per user
		Map<Integer,Set<Integer>> userTrainItems=new HashMap<Integer,Set<Integer>>();
		for(int[] r:train){
		Set<Integer> s=userTrainItems.get(r[0]);
		if(s==null){s=new HashSet<Integer>();userTrainItems.put(r[0],s);}
		s.add(r[1]);
		}
		
		// All items
		Set<Integer> allItemSet=new HashSet<Integer>();
		for(int[] r:train) allItemSet.add(r[1]);
		for(int[] r:test) allItemSet.add(r[1]);
		
		// Shapley + Sugeno
		float[] shapley = computeShapley(itemIdx);
		double[] g = new double[N_DIMS];
		float sumS=0;for(int d=0;d<N_DIMS;d++) sumS+=Math.max(0,shapley[d]);
		for(int d=0;d<N_DIMS;d++) g[d]=(sumS>0)?Math.max(0.001,shapley[d])/sumS:1.0/N_DIMS;
		
		// === MAE + RMSE ===
		float sumAbs=0, sumSq=0; int nPred=0;
		for(int[] tst:test){
		float pred=predictMAE(tst,itemIdx,uMeans,gMean,g);
		sumAbs+=Math.abs(pred-tst[2]); sumSq+=(pred-tst[2])*(pred-tst[2]); nPred++;
		}
		float mae=(nPred>0)?sumAbs/nPred:0;
		float rmse=(nPred>0)?(float)Math.sqrt(sumSq/nPred):0;
		
		// === RANKING ===
		Map<Integer,ArrayList<int[]>> userTest=new HashMap<Integer,ArrayList<int[]>>();
		for(int[] t:test){
		ArrayList<int[]> l=userTest.get(t[0]);
		if(l==null){l=new ArrayList<int[]>();userTest.put(t[0],l);}
		l.add(t);
		}
		
		double sP5=0,sP10=0,sR5=0,sR10=0,sN5=0,sN10=0;
		int nUAll=0,nURel=0;
		
		
		for(Map.Entry<Integer,ArrayList<int[]>> ue:userTest.entrySet()){
            int uid=ue.getKey();
            ArrayList<int[]> testSits=ue.getValue();

            // Aggregate test items
            Map<Integer,float[]> itemPreds=new HashMap<Integer,float[]>();
            for(int[] t:testSits){
                float pred=predictMAE(t,itemIdx,uMeans,gMean,g);
                float[] acc=itemPreds.get(t[1]);
                if(acc==null){acc=new float[]{0,0,t[2]};itemPreds.put(t[1],acc);}
                acc[0]+=pred; acc[1]++;
                if(t[2]>acc[2]) acc[2]=t[2];
            }

            Set<Integer> relSet=new HashSet<Integer>();
            for(Map.Entry<Integer,float[]> ie:itemPreds.entrySet())
                if(ie.getValue()[2]>=4) relSet.add(ie.getKey());

            Set<Integer> trainItems=userTrainItems.containsKey(uid)?
                userTrainItems.get(uid):new HashSet<Integer>();

            // Rank ALL 20 items
            float uMean2=uMeans.containsKey(uid)?uMeans.get(uid):gMean;
            int[] userCtx=new int[N_DIMS];
            for(int d=0;d<N_DIMS;d++) userCtx[d]=testSits.get(0)[3+d];

            ArrayList<float[]> scored=new ArrayList<float[]>();
            for(int iid:allItemSet){
                float pred;
                if(itemPreds.containsKey(iid)){
                    // Test item: predicted average
                    pred=itemPreds.get(iid)[0]/itemPreds.get(iid)[1];
                } else if(trainItems.contains(iid)){
                    // Training item: actual rating average (use training data)
                    pred=uMean2; // fallback
                } else {
                    // Unrated item: predict with context
                    pred=predictRank(uid,iid,userCtx,itemIdx,uMeans,gMean,g);
                }
                scored.add(new float[]{iid,pred,relSet.contains(iid)?1:0});
            }

            Collections.sort(scored,new Comparator<float[]>(){
                public int compare(float[] a,float[] b){return Float.compare(b[1],a[1]);}
            });

            /*nUAll++;
            sP5+=precAtK(scored,relSet,5);
            sP10+=precAtK(scored,relSet,10);

            if(!relSet.isEmpty()){
                sR5+=recAtK(scored,relSet,5);
                sR10+=recAtK(scored,relSet,10);
                sN5+=ndcgAtK(scored,relSet,5);
                sN10+=ndcgAtK(scored,relSet,10);
                nURel++;
            }*/
            /*nUAll++;
            sP5+=precAtK(scored,relSet,5);
            sP10+=precAtK(scored,relSet,10);
            sR5+=(relSet.isEmpty()?0:recAtK(scored,relSet,5));
            sR10+=(relSet.isEmpty()?0:recAtK(scored,relSet,10));
            sN5+=ndcgAtK(scored,relSet,5);
            sN10+=ndcgAtK(scored,relSet,10);*/
            // Plus de nURel, tout sur nUAll
            nUAll++;
            sP5+=precAtK(scored,relSet,5);
            sP10+=precAtK(scored,relSet,10);
            if(!relSet.isEmpty()){
                sR5+=recAtK(scored,relSet,5);
                sR10+=recAtK(scored,relSet,10);
                sN5+=ndcgAtK(scored,relSet,5);
                sN10+=ndcgAtK(scored,relSet,10);
                nURel++;
            }
            
            
            
            
            
        }
		
		
		
		

		if(nUAll==0)nUAll=1;if(nURel==0)nURel=1;
		/*lastMetrics=new double[]{mae,rmse,
	            sP5/nUAll,sP10/nUAll,
	            sR5/nUAll,sR10/nUAll,
	            sN5/nUAll,sN10/nUAll};*/
		lastMetrics=new double[]{mae,rmse,
	            sP5/nUAll, sP10/nUAll,       // Prec: nUAll
	            sR5/nURel, sR10/nURel,       // Rec: nURel
	            sN5/nURel, sN10/nURel};      // NDCG: nURel
		
		System.out.printf("SEED=%d FOLD=%d MAE=%.4f RMSE=%.4f P@5=%.3f R@5=%.3f N@5=%.3f%n",
		seed,fold,mae,rmse,(float)(sP5/nUAll),(float)(sR5/nURel),(float)(sN5/nURel));
		}
		
		// Predict for MAE (context from test item)
		private static float predictMAE(int[] tst,
		Map<Integer,ArrayList<int[]>> itemIdx,
		Map<Integer,Float> uMeans,float gMean,double[] g){
		int uid=tst[0],iid=tst[1];
		float uMean=uMeans.containsKey(uid)?uMeans.get(uid):gMean;
		ArrayList<int[]> neighbors=itemIdx.get(iid);
		if(neighbors==null||neighbors.isEmpty()) return Math.max(1,Math.min(5,uMean));
		float sumW=0,sumWR=0;int numN=0;
		for(int[] nb:neighbors){
		if(nb[0]==uid)continue;
		double[] fd=new double[N_DIMS];
		for(int d=0;d<N_DIMS;d++){
		int tv=tst[3+d],nv=nb[3+d];
		fd[d]=(tv>0&&nv>0&&tv==nv)?1.0:0.0;
		}
		float choquet=choquetIntegral(fd,g);
		if(choquet<0.001f)continue;
		float weight=choquet*choquet*choquet;
		float nMean=uMeans.containsKey(nb[0])?uMeans.get(nb[0]):gMean;
		sumWR+=weight*(nb[2]-nMean);sumW+=Math.abs(weight);numN++;
		}
		if(numN>0&&sumW>0) return Math.max(1,Math.min(5,uMean+sumWR/sumW));
		return Math.max(1,Math.min(5,uMean));
		}
		
		// Predict for ranking (context from user)
		private static float predictRank(int uid,int iid,int[] userCtx,
		Map<Integer,ArrayList<int[]>> itemIdx,
		Map<Integer,Float> uMeans,float gMean,double[] g){
		float uMean=uMeans.containsKey(uid)?uMeans.get(uid):gMean;
		ArrayList<int[]> neighbors=itemIdx.get(iid);
		if(neighbors==null||neighbors.isEmpty()) return uMean;
		float sumW=0,sumWR=0;int numN=0;
		for(int[] nb:neighbors){
		if(nb[0]==uid)continue;
		double[] fd=new double[N_DIMS];
		for(int d=0;d<N_DIMS;d++){
		int tv=userCtx[d],nv=nb[3+d];
		fd[d]=(tv>0&&nv>0&&tv==nv)?1.0:0.0;
		}
		float choquet=choquetIntegral(fd,g);
		if(choquet<0.001f)continue;
		float weight=choquet*choquet*choquet;
		float nMean=uMeans.containsKey(nb[0])?uMeans.get(nb[0]):gMean;
		sumWR+=weight*(nb[2]-nMean);sumW+=Math.abs(weight);numN++;
		}
		if(numN>0&&sumW>0) return Math.max(1,Math.min(5,uMean+sumWR/sumW));
		return uMean;
		}

    // Shapley
    private static float[] computeShapley(Map<Integer,ArrayList<int[]>> itemIdx) {
        double sB=0; long cnt=0;
        double[] mE=new double[N_DIMS], mC=new double[N_DIMS];
        for(ArrayList<int[]> sits : itemIdx.values()){
            int sz=sits.size(); if(sz<2) continue;
            for(int i=0;i<sz;i++) for(int j=i+1;j<sz;j++){
                int[] a=sits.get(i), b=sits.get(j);
                if(a[0]==b[0]) continue;
                double diff=Math.abs(a[2]-b[2]); sB+=diff; cnt++;
                for(int d=0;d<N_DIMS;d++)
                    if(a[3+d]>0&&b[3+d]>0&&a[3+d]==b[3+d]){mE[d]+=diff;mC[d]++;}
            }
        }
        float[] w=new float[N_DIMS];
        if(cnt==0){Arrays.fill(w,1f/N_DIMS);return w;}
        double base=sB/cnt, maxS=0;
        double[] sh=new double[N_DIMS];
        for(int d=0;d<N_DIMS;d++){
            if(mC[d]>5) sh[d]=base-mE[d]/mC[d];
            if(sh[d]>maxS) maxS=sh[d];
        }
        for(int d=0;d<N_DIMS;d++)
            w[d]=(maxS>0)?(float)Math.max(0.01,sh[d]/maxS):1f/N_DIMS;
        return w;
    }
    // metrics
    
    
    static double precAtK(ArrayList<float[]> r,Set<Integer> rel,int k){
        int h=0;for(int i=0;i<Math.min(k,r.size());i++)if(rel.contains((int)r.get(i)[0]))h++;return(double)h/k;}
    static double recAtK(ArrayList<float[]> r,Set<Integer> rel,int k){
        if(rel.isEmpty())return 0;int h=0;for(int i=0;i<Math.min(k,r.size());i++)if(rel.contains((int)r.get(i)[0]))h++;return(double)h/rel.size();}
    static double ndcgAtK(ArrayList<float[]> r,Set<Integer> rel,int k){
        double dcg=0;for(int i=0;i<Math.min(k,r.size());i++)if(rel.contains((int)r.get(i)[0]))dcg+=1.0/(Math.log(i+2)/Math.log(2));
        double idcg=0;for(int i=0;i<Math.min(k,rel.size());i++)idcg+=1.0/(Math.log(i+2)/Math.log(2));return(idcg>0)?dcg/idcg:0;}

    // Choquet + Sugeno
    private static float choquetIntegral(double[] v, double[] g) {
        int n=v.length; if(n==0) return 0;
        double lam=sugenoLambda(g);
        int[] idx=new int[n]; for(int i=0;i<n;i++) idx[i]=i;
        for(int i=1;i<n;i++){int k=idx[i];int j=i-1;while(j>=0&&v[idx[j]]>v[k]){idx[j+1]=idx[j];j--;}idx[j+1]=k;}
        double ch=0,prev=0;
        for(int i=0;i<n;i++){double ai=v[idx[i]];if(ai<=prev){prev=ai;continue;}
            int mask=0;for(int k=i;k<n;k++)mask|=(1<<idx[k]);
            ch+=(ai-prev)*sugenoMu(mask,g,lam);prev=ai;}
        return (float)Math.max(0,Math.min(1,ch));
    }
    private static double sugenoLambda(double[] g){
        double s=0;for(double gi:g)s+=gi;if(Math.abs(s-1)<1e-6)return 0;
        double lo,hi;if(s>1){lo=-1+1e-6;hi=0;}else{lo=0;hi=50;}
        for(int i=0;i<100;i++){double mid=(lo+hi)/2,prod=1;for(double gi:g)prod*=(1+mid*gi);
            if(Math.abs(prod-1-mid)<1e-10)return mid;if(prod-1-mid>0)hi=mid;else lo=mid;}
        return(lo+hi)/2;
    }
    private static double sugenoMu(int mask,double[] g,double lam){
        if(mask==0)return 0;if(Math.abs(lam)<1e-10){double s=0;for(int i=0;i<g.length;i++)if((mask&(1<<i))!=0)s+=g[i];return Math.min(1,s);}
        double prod=1;for(int i=0;i<g.length;i++)if((mask&(1<<i))!=0)prod*=(1+lam*g[i]);
        return Math.min(1,Math.max(0,(prod-1)/lam));
    }

    private static double mean(ArrayList<Double> v){double s=0;for(double x:v)s+=x;return s/v.size();}
    private static double std(ArrayList<Double> v,double m){double s=0;for(double x:v)s+=(x-m)*(x-m);return Math.sqrt(s/v.size());}
}
