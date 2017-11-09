package perceptron;

import java.util.Arrays;





public class OnlinePerceptron {
    public static final int DIM = 3; // dimension de l'espace de repr�sentation
    public static float[] w = new float[DIM]; // param�tres du mod�le
    public static float[][] data = { // les observations
    	{1,0,0}, {1,0,1} , {1,1,0},
    	{1,1,1}
    }; 
    public static int[] refs = {-1, -1, -1, 1};//, 1, 1, 1}; // les r�ferences

    


    

    public static int epoque(float[][] corpus, int[] labels, float[] model){

        int nbUpdate = 0;

        for (int i = 0; i < corpus.length; i++) {

            nbUpdate = online_step(corpus[i],labels[i],model);

        }

        return nbUpdate;
}
    
    public static void UpdateW(float[] corpus, int lab, float[] models, float lr) {

        // Boucle

        for (int i = 0; i < corpus.length; i++){

            models[i] = models[i] + corpus[i] * lab * lr;

        }
}

    
       
 // Test des données

    public static int online_step(float[] obs, int ref, float[] model) {

        int err = 0;

        float lr = 1;

        float signe = dot(obs, model);

        if (ref * signe <= 0) {

            UpdateW(obs, ref, model, lr);

            err++;

            return err;

        } else return 0;
}
	
    /* Produit Scalaire */

    public  static float dot(float[] x, float[] y) {

        float res = x[0]*y[0]+x[1]*y[1]+x[2]*y[2];

        return res;
}

    
    public static void main(String[] args) {

        final int EPOCHMAX = 20;

        System.out.println("w= "+Arrays.toString(w));

   /* lets go */

        for (int i = 0; i < EPOCHMAX; i++) {

            epoque(data,refs,w);

            System.out.println("w= "+Arrays.toString(w));

    }
}
}