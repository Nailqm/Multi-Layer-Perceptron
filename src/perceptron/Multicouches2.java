package perceptron;

import java.util.ArrayList;
import java.util.Random;

import image.ImageConverter;
import mnisttools.MnistReader;


public class Multicouches2 {

	/* Les donnees */
	public static String path = "..\\Multi Layer\\"; //TODO change the path
	public static String labelDB = path + "train-labels-idx1-ubyte";
	public static String imageDB = path + "train-images-idx3-ubyte";

	/* Parametres */
	// les N premiers exemples pour l'apprentissage
	public static final int N = 100;
	// les T derniers exemples pour l'evaluation
	public static final int T = 100;
	// Nombre d'epoque max
	public final static int EPOCHMAX = 20;
	// Learning rate
	public static float lr = (float) 0.001;
	public static int s_batch = 20;

	public static void Load(ArrayList<float[]> trainData, ArrayList<Integer> trainRefs, ArrayList<float[]> testData,
			ArrayList<Integer> testRefs) {
		System.err.println("# Load the database !");
		/* Lecteur d'image */
		MnistReader db = new MnistReader(labelDB, imageDB);
		/* Taille des images et donc de l'espace de representation */
		final int SIZEW = ImageConverter.image2VecteurReel_withB(db.getImage(1)).length;

		/* Creation des donnees */
		/* Donnees d'apprentissage */
		for (int i = 1; i <= N; i++) {
			trainData.add(ImageConverter.image2VecteurReel_withB(db.getImage(i)));
			trainRefs.add(db.getLabel(i));
		}

		for (int i = N + 1; i <= N + T ; i++) {
			testData.add(ImageConverter.image2VecteurReel_withB(db.getImage(i)));
			testRefs.add(db.getLabel(i));
		}
	}

	// update for the last layer : z_i^(2)(z_j^(3) - p)=gradient
	public static void MAJ_l3(float[][] w, float[][] delta, float[][] label, float[][] z_3, float[][] z_2) {
		for (int n = 0; n < label.length; n++) {
			for (int j = 0; j < w.length; j++) {
				for (int i = 0; i < w[j].length; i++) {
					w[j][i] = w[j][i] + z_2[n][i] * delta[n][j] * lr;
				}
			}
		}
	}

	// Compute the term (z_j^(3) - p)=delta
	public static float[][] ComputeDelta_l3(float[][] w, float[][] label, float[][] z_3) {
		float[][] delta = new float[label.length][w.length];
		for (int n = 0; n < label.length; n++) {
			for (int j = 0; j < w.length; j++) {
				delta[n][j] = z_3[n][j] - label[n][j];
				//System.out.print(delta[n][j]+"\n");
			}
		}
		return delta;
	}

	// update for the first layer : gradient = z_i^(1)f_2'(u_j^(2)) \sum_k wjk^(3) \delta_k^(3)
	public static void MAJ_l2(float[][] w2, float[][] w3, float[][] label, float[][] z_2, float[][] u_2, float[][] z_1,
			float[][] delta) {
		float gradient = 0;
		float sum = 0;
		

		for (int n = 0; n < label.length; n++) {
			for (int j = 0; j < w2.length; j++) {
				for (int i = 0; i < w2[j].length; i++) {
					sum = 0;
					for (int c = 0; c < w3.length; c++) {
						sum += w3[c][j] *delta[n][c];
					}
					// System.out.print(i);
					// System.out.print(j);
					gradient = z_1[n][i] * d_sig(u_2[n][j]) * sum;
					w2[j][i] = w2[j][i] + lr * gradient;
				}
			}
		}
	}
	


	public static float softmax(float[] u_q, float u_j) {
		float sum = 0;
		float max = 0;
		for (int i=0; i<u_q.length; i++) {
			if (max<=u_q[i]){
				max=u_q[i];
			}
		}
		
		for (int i = 0; i < u_q.length; i++) {
			sum += Math.exp(u_q[i]);
		}
		return (float) (Math.exp(u_j)) / sum;
	}

	// compute u = x1w1 + x2w2 +......
	public static float compute_u(float[] w, float[] x) {
		float res = 0;
		for (int i = 0; i < x.length; i++) {
			res += w[i] * x[i];
		}
		// System.out.print(w[w.length-1]);
		return res + w[w.length - 1];
	}
	
	//compute all the u for a layer
	public static float[] uLayer3(float[][] w, float[] x) {

		float[] res = new float[w.length];
		for (int j = 0; j < w.length; j++) {
			res[j] = compute_u(w[j], x);
		}
		return res;
	}
	
	//compute all the u for a layer
	public static float[] uLayer2(float[][] w, float[] x) {

		float[] res = new float[w.length];
		for (int j = 0; j < w.length; j++) {
			res[j] = compute_u(w[j], x);
		}
		// System.out.print(res[res.length-1]);
		return res;
	}

	// Compute neurons output for all the batch
	public static ArrayList<float[][]> ComputeOut(ArrayList<float[][]> w, ArrayList<float[]> data) {

		ArrayList<float[][]> z = new ArrayList<float[][]>();

		float[][] u_2 = new float[data.size()][w.get(0).length - 1];
		/**
		 * for (int i=0; i<u_2[0].length;i++) {
		 * System.out.print(u_2[0][i]+"\n"); }
		 **/
		for (int n = 0; n < data.size(); n++) {
			u_2[n] = uLayer2(w.get(0), data.get(n));
		}
		
			/**
	    for (int i=0; i<u_2[n].length;i++) {
			 System.out.print(u_2[n][i]+"\n"); } 
		}
		**/
		// System.out.print("u_2 " + u_2[0].length );

		float[][] z_2 = new float[data.size()][w.get(0).length];
		for (int n = 0; n < data.size(); n++) {
			for (int j = 0; j < w.get(0).length - 1; j++) {
				z_2[n][j] = sigmoid(u_2[n][j]); // problem
				//System.out.print(z_2[n][j] + "\n");
			}
			z_2[n][w.get(0).length - 1] = 1;
		}
		// System.out.print("z_2 " + z_2[0].length );

		float[][] u_3 = new float[data.size()][w.get(1).length];
		for (int n = 0; n < data.size(); n++) {
			u_3[n] = uLayer3(w.get(1), z_2[n]);
			/**
			 for (int i=0; i<u_3[n].length;i++) {
			  System.out.print(u_3[n][i]+"\n"); }
		**/	 
		}
		

		float[][] z_3 = new float[data.size()][w.get(1).length];
		for (int n = 0; n < data.size(); n++) {
			for (int j = 0; j < w.get(1).length; j++) {
				z_3[n][j] = softmax(u_3[n], u_3[n][j]);
				//System.out.print(z_3[n][j]+"\n");
			}
		}

		z.add(u_2);
		z.add(z_2);
		z.add(u_3);
		z.add(z_3);

		// z will contain u_j and z_j for the two layers
		return z;

	}

	/**
	 * for (int n=0; n<data.size(); n++) {
	 * 
	 * res[0] = uLayer (w.get(0), data.get(n)); // u_2
	 * 
	 * for (int i = 0; i<w.get(0).length; i++) { res[1][i] = sigmoid
	 * (compute_u(w.get(0)[i], data.get(n))); } // z_2
	 * 
	 * res[2] = uLayer(w.get(1), res[1]); // u_3 for (int i = 0;
	 * i<w.get(1).length; i++) { res[3][i] = softmax(res[2],
	 * compute_u(w.get(1)[i], res[1]) ); } // z_3
	 * 
	 * z.add(res); }
	 **/

	public static float sigmoid(float x) {
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}

	public static float d_sig(float x) {
		return (float) sigmoid(x) * (1 - sigmoid(x));
	}
    
	//not used
	public static float dot(float[] x1, float[] x2) {
		float res = 0;
		for (int i = 0; i < x1.length; i++)
			res += x1[i] * x2[i];
		return res;
	}

	
	//initialize all the w
	public static ArrayList<float[][]> CreateModel(int N1, int N2, int N3) {
		// return all the weight matrices initialized
		ArrayList<float[][]> w = new ArrayList<float[][]>();

		float w2[][] = new float[N2 + 1][N1 + 1];
		float w3[][] = new float[N3][N2 + 1];
		for (int j = 0; j < N2 + 1; j++) {
			for (int i = 0; i < N1 + 1; i++) {
				 w2[j][i]= (float) (1+Math.random()*9)/10;
				//w2[j][i] = ((float) (2 * Math.random() - 1)) / 100;
				//System.out.print(w2[j][i]+"\n");
			}
		}
		for (int j = 0; j < N3; j++) {
			for (int i = 0; i < N2 + 1; i++) {
				w3[j][i]=(float) (1+Math.random()*9)/10;
				//w3[j][i] = ((float) (2 * Math.random() - 1)) / 100;
				//System.out.print(w3[j][i]+"\n");
			}
		}
		w.add(w2);
		w.add(w3);

		return w;
	}

	//transform a reference into a vector of label which contain 0 and 1
	public static float[] refsToLabel(int ref, int nbSorties) {
		float[] res = new float[nbSorties];
		for (int i = 0; i < nbSorties; i++) {
			if (i == ref) {
				res[i] = 1;
			} else {
				res[i] = 0;
			}
		}
		return res;
	}


	
	//update 
	public static void fit(ArrayList<float[]> trainData, ArrayList<Integer> trainRefs, ArrayList<float[][]> w,
			int niter) {

		float label[][] = new float[trainData.size()][w.get(1).length];
		for (int n = 0; n < trainData.size(); n++) {
			label[n] = refsToLabel(trainRefs.get(n), w.get(1).length);
		}

		ArrayList<float[][]> outs = new ArrayList<float[][]>();
		float[][] z_1 = new float[trainData.size()][trainData.get(0).length + 1]; 
		//get inputs of all the training samples
		
		float u_2[][] = new float [trainData.size()][w.get(0).length];
		//get inputs of layer 2 of all the training samples
		float z_2[][] = new float [trainData.size()][w.get(0).length];
		//get outputs of layer 2 of all the training samples
		float z_3[][] = new float [trainData.size()][w.get(1).length];
		//get outputs of layers 3 of all the training samples
		
		for (int m = 0; m < niter; m++) {
			outs = ComputeOut(w, trainData);
			u_2 = outs.get(0);
			z_2 = outs.get(1);
			z_3 = outs.get(3);

			for (int n = 0; n < trainData.size(); n++) {
				for (int i = 0; i < trainData.get(0).length; i++) {
					z_1[n][i] = trainData.get(n)[i];
				}
				z_1[n][trainData.get(0).length] = 1;
			}

			float delta[][] = ComputeDelta_l3(w.get(1), label, z_3);

			MAJ_l3(w.get(1), delta, label, z_3, z_2);
			MAJ_l2(w.get(0), w.get(1), label, z_2, u_2, z_1, delta);
		}
		// update the w online (image by image) or by batch

	}

	public static void test(ArrayList<float[]> testData, ArrayList<float[][]> w, ArrayList<Integer> testRefs) {
		ArrayList<float[][]> outs = ComputeOut(w, testData);
		float max = 0;
		int index = 0;
		int countErrors = 0;
		int countSuccess = 0;
		float[][] z_3 = outs.get(3);
		for (int n = 0; n < 100; n++) {
			for (int j = 0; j < z_3[n].length; j++) {
				System.out.print(z_3[n][j]+"\n");
				if (max < z_3[n][j]) {
					max = z_3[n][j];
					index = j;            //checked
				}
				//System.out.print(index+"\n");
				//System.out.print(testRefs.get(n)+"---"+"\n");
				
			}
			if (index == testRefs.get(n)) {
				countSuccess++;
			} else {
				countErrors++;
			}
		}
		System.out.print("Errors: " + countErrors);
		System.out.print("Success: " + countSuccess);
		System.out.print("Success rate = " + ((float) countSuccess) / testData.size());

	}

	/**
	  public static void fit_batch(ArrayList<float[]> trainData, float[][] label, ArrayList<float[][]> w, int n) {
		  float [][] z_1 = new float [trainData.size()/n][trainData.get(0).length];
	  
	      for (int i=0; i<trainData.size(); i++) { 
	    	  z_1[i] = trainData.get(i); 
	    	  } 
	      //convert trainData to type float[][]
	  
	      ArrayList<float[][]> outs = ComputeOut(w, trainData); 
	      float u_2[][] = outs.get(0); 
	      float z_2[][] = outs.get(1); float u_3[][] = outs.get(2);
	      float z_3[][] = outs.get(3);
	  
	      float delta [][] = ComputeDelta_l3 (w.get(1), label, z_3);
	  
	      MAJ_l3(w.get(1),delta,label,z_3, z_2); 
	      MAJ_l2(w.get(0), w.get(1), label, z_2, u_2, z_1, delta);
	  }
	  // update the w's for an image or a whole mini-batch }
	**/

	public static void main(String[] args) {
		System.out.println("Start ...");
		ArrayList<float[]> trainData = new ArrayList<float[]>();
		ArrayList<float[]> testData = new ArrayList<float[]>();
		ArrayList<Integer> trainRefs = new ArrayList<Integer>();
		ArrayList<Integer> testRefs = new ArrayList<Integer>();
		Load(trainData, trainRefs, testData, testRefs);

		for (int n = 0; n < trainData.size(); n++) {
			for (int i = 0; i < trainData.get(n).length; i++) {
				trainData.get(n)[i] = trainData.get(n)[i];
			}
		}

		System.out.println("Size of training " + trainData.size());
		int N1 = trainData.get(0).length;
		// System.out.print(N1);
		int N2 = 50;
		int N3 = 10;
		System.out.print(trainRefs.get(0) + "\n");

		// System.out.print(sigmoid((float)0.6));
		/**
		 * for (int i=2; i<3; i++) { System.out.print(trainRefs.get(i)); float[]
		 * t= refsToLabel(trainRefs.get(i),N3); for (float l:t) {
		 * System.out.print(l); } }
		 * 
		 **/

		ArrayList<float[][]> w = CreateModel(N1, N2, N3);
		int count = 0;
		/**
		 * for (int i=0; i<trainData.get(0).length;i++) {
		 * System.out.print(trainData.get(0)[i]+"\n"); count++; }
		 **/
		System.out.print(count);
		System.out.println("Number of hidden layers " + w.size());
		fit(trainData, trainRefs, w, 10);
		/**
		for (int j = 0; j < w.get(0).length; j++) {
			for (int i = 0; i < w.get(0)[j].length; i++) {
				// w2[j][i]= (float) (1+Math.random()*9)/10;
				
				System.out.print(w.get(0)[j][i]+"\n");
			}
		}
		
		System.out.print("----------------------------------");
		
		for (int j = 0; j < w.get(1).length; j++) {
			for (int i = 0; i < w.get(1)[j].length; i++) {
				// w2[j][i]= (float) (1+Math.random()*9)/10;
				
				System.out.print(w.get(1)[j][i]+"\n");
			}
		}
		**/
		test(testData, w, testRefs);
		

	}

}