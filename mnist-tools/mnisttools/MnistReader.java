package mnisttools;

import java.io.IOException;

import mnisttools.MnistManager;

/**
 * 
 * @author allauzen
 * This class wrap the MNIST tools to avoid exception handling 
 * and to provide an easy access to the data.
 * 
 */
public class MnistReader {

	/*File name for label and image DB*/
	private String labelDB;
	private String imageDB;
	private int totalImages;
	private MnistManager manager=null;


	public MnistReader(String labelDB, String imageDB) {
		super();
		this.labelDB = labelDB;
		this.imageDB = imageDB;
		/* Open databases */
		try {
			this.manager = new MnistManager(imageDB,labelDB);
		} catch (IOException e) {
			e.printStackTrace();
			System.err.println("Maybe one of the specified files does not exist !");
			System.err.println("Check:\n\t"+this.imageDB+"\n\t"+this.labelDB);
			System.exit(2);
		}
		this.totalImages = this.manager.getImages().getCount();
	}

	public int[][] getImage(int idx){
		int[][] image=null;
		this.manager.setCurrent(idx);
		try {
			image = manager.readImage();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(2);
		}
		return image;
	}

	public int getLabel(int idx){
		int label=-1;
		this.manager.setCurrent(idx);
		try {
			label = manager.readLabel();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(2);
		}
		return label;
	}

	public int getTotalImages() {
		return totalImages;
	}

}
