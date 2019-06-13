
package boucoiran.fr.gridgrabber;

import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import static org.opencv.core.CvType.CV_8UC1;

public class MainActivity extends Activity {
    private static final String  TAG = "MainActivity";
    private int step = 0;
    private static int RESULT_LOAD_IMG = 1;
    private static Mat mat2 = null;
    String imgDecodableString;

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }


    @Override
    @TargetApi(23)
    protected void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        if (!OpenCVLoader.initDebug()) {
            Log.i(TAG, "onCreate: could NOT load OPen CV");
        } else {
            Log.i(TAG, "onCreate: OPENCV Loaded successfully");
        }

        setContentView(R.layout.activity_main);

        final Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                doNext();
            }
        });

        final Button SaveButton = (Button) findViewById(R.id.SaveButton);
        SaveButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                saveFile();
            }
        });
        
        ImageView imageView = findViewById(R.id.imageView);
        loadImagefromGallery(imageView);

        /*
        Log.i(TAG, "onCreate: trying to get a Mat from an image");

        try{
            mat2 = Utils.loadResource(this, R.drawable.grid2, CvType.CV_8UC1);
            //mat2 = Utils.loadResource(this, R.drawable.simpleredrect, CvType.CV_8UC1);
        } catch (Exception e) {
            Log.i(TAG, "onCreate: "+e.getStackTrace());
        }
        if (mat2 == null) {
            Log.i(TAG, "onCreate: image NOT loaded");
        } else {
            Log.i(TAG, "onCreate: image Loaded SUCCESSFULLY");
        }
        */
    }

    private void doNext() {
        Log.i(TAG, "doNext: step = "+step);
        TextView stepTV= (TextView)findViewById(R.id.stepView);
        switch(step) {
            case 1:
                displayImageFromMat(mat2);
                stepTV.setText("1. Just Displaying");
                break;
            case 2:
                doGaussianBlur(mat2);
                stepTV.setText("2. Gaussian Blur");
                break;
            case 3:
                doAdaptiveThreshold(mat2);
                stepTV.setText("3. Adaptive Threshold");
                break;
            case 4:
                stepTV.setText("4. Reversing Image");
                reverseImage();
                break;
            case 5:
                stepTV.setText("5. Dilating");
                dilateImage();
                break;
            case 6:
                stepTV.setText("6. detecting blobs");
                detectBlobs();
                break;
            case 7:
                stepTV.setText("7. eroding image");
                erodeImage();
                break;
            case 8:
                stepTV.setText("8. Find lines");
                findLines();
                break;
            case 100:
                stepTV.setText("100. Just display");
                displayImageFromMat(mat2);
                break;
            case 101:
                stepTV.setText("100. simple Flood Fill");
                simpleFloodFill();
                break;


        }
        step+=1;
    }

    private void findLines() {
        Log.i(TAG, "findLines: starting Hough Transform");
        /*
        Mat lines = new Mat();
        Imgproc.HoughLines(mat2, lines, 1, Math.PI/180, 200);

        for (int i = 0; i < lines.cols(); i++) {
            double[] val = lines.get(0, i);
            if(val[1] != 0) {
                double m = -1/Math.tan(val[1]);
                double c = val[0]/Math.sin(val[1]);
                Imgproc.line(mat2, new Point(0, c), new Point(mat2.size().width, m*mat2.size().width+c), new Scalar(255));
            } else {
                Imgproc.line(mat2, new Point(val[0], 0), new Point(val[0], mat2.size().height), new Scalar(255), 112);
            }
            Imgproc.line(mat2, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(255), 12);
        }

        // Show results
        */
        Mat houghTransMat = getHoughPTransform(mat2, 1, Math.PI/180, 200);
        Log.i(TAG, "findLines: finished Hough Transform");
        displayImageFromMat(houghTransMat);

    }

    public Mat getHoughPTransform(Mat image, double rho, double theta, int threshold) {
        Mat lines = new Mat();
        Imgproc.HoughLinesP(image, lines, rho, theta, threshold);

        for (int i = 0; i < lines.cols(); i++) {
            double[] val = lines.get(0, i);
            Imgproc.line(mat2, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(0, 0, 125), 4);
        }
        return mat2;
    }

    private void erodeImage() {
        Mat kernel = new Mat(3, 3, CvType.CV_8UC1);
        kernel.put(0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0);
        Imgproc.morphologyEx(mat2, mat2, Imgproc.MORPH_ERODE, kernel);
        displayImageFromMat(mat2);
    }

    private void simpleFloodFill() {

        int w = mat2.width();
        int h = mat2.height();
        Mat floodFilled = new Mat(mat2.height()+2, mat2.width()+2, CV_8UC1);
        Scalar colorG = new Scalar(200);

        //byte buff[] = new byte[(int)mat2.total() * (int)mat2.channels()];
        //mat2.get(0, 0, buff);
        //displayImageFromMat(mat2);
        Imgproc.floodFill(mat2, floodFilled, new Point(10,10), colorG);

        displayImageFromMat(mat2);
    }


    @TargetApi(23)
    private void saveFile() {
        //Imgproc.cvtColor(mat2, mat2, CvType.CV_8UC1);
        //String path = Environment.getExternalStorageDirectory().getAbsolutePath().toLowerCase();
        Log.i(TAG, "saveFile:  saving ");


        int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }



        try {
            boolean imwrite = Imgcodecs.imwrite("test.bmp", mat2);
            Log.i(TAG, "saveFile: imwrite " + imwrite);
            String root = Environment.getExternalStorageDirectory().toString();
            File myDir = new File(root + "/saved_images");
            Log.i("Bitmap", root);
            myDir.mkdirs();
            //int n = i;
            String path = Environment.getExternalStorageDirectory().toString();

            OutputStream fOut = null;
            File file = new File(this.getFilesDir(), "/image1.jpg");
            fOut = new FileOutputStream(file);

            Bitmap bm = Bitmap.createBitmap((int) mat2.size().width, (int) mat2.size().height, Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(mat2, bm);
            bm.compress(Bitmap.CompressFormat.JPEG, 85, fOut);
            fOut.flush();
            fOut.close();

            MediaStore.Images.Media.insertImage(getContentResolver()
                    , file.getAbsolutePath(), file.getName(), file.getName());
        } catch (Exception e) {
            Log.i(TAG, "saveFile: exception caught " + e.getMessage());
        }
    }

    private void detectBlobs() {
        Log.i(TAG, "detectBlobs: Starting detection");
        //Mat floodFilled = new Mat(mat2.height()+2, mat2.width()+2, CV_8UC1);
        Mat floodFilled = new Mat();
        int w = mat2.width();
        int h = mat2.height();

        int max=-1;
        Scalar colorG = new Scalar(64);
        Scalar colorBlack = new Scalar(35);

        byte buff[] = new byte[(int)mat2.total() * (int)mat2.channels()];
        mat2.get(0, 0, buff);

        Point maxPt = new Point(0,0);

        for(int y=0;y<h;y++)
        {
            if(y%1000 == 0) {
                Log.i(TAG, "detectBlobs: update line " + y + " of "+h);
            }
            Mat row = mat2.row(y);
            byte rowBuff[] = new byte[(int)mat2.width()* (int)mat2.channels()];
            row.get(0,0,rowBuff);
            for(int x=0;x<w;x++)
            {
                if((int)(rowBuff[x] &0xFF) >= 128) {
                    int area = Imgproc.floodFill(mat2, floodFilled, new Point(x,y), colorG);
                    if(area>max)
                    {
                        Log.i(TAG, "detectBlobs: updating max");
                        maxPt = new Point(x,y);
                        max = area;
                    }
                }

            }
        }
        Log.i(TAG, "detectBlobs: Finished detection");
        Log.i(TAG, "detectBlobs: Filling in smaller blobs");

        //make our largest blob white
        int area2 = Imgproc.floodFill(mat2, floodFilled, maxPt, new Scalar(255));

        for(int y=0;y<h;y++)
        {
            if(y%1000 == 0) {
                Log.i(TAG, "detectBlobs: update line " + y + " of "+h);
            }
            Mat row = mat2.row(y);
            byte rowBuff[] = new byte[(int)mat2.width()* (int)mat2.channels()];
            row.get(0,0,rowBuff);
            for(int x=0;x<w;x++)
            {
                if((int)(rowBuff[x] & 0xFF)==64 && x!=maxPt.x && y!=maxPt.y)
                {
                    //fill all smaller blobs to black
                    int area = Imgproc.floodFill(mat2, floodFilled, new Point(x,y), colorBlack);
                }
            }
        }
        Log.i(TAG, "detectBlobs: FINISHED Filling in smaller blobs");
        displayImageFromMat(mat2);
    }

    private void dilateImage() {
        Mat kernel = new Mat(3, 3, CvType.CV_8UC1);
        kernel.put(0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0);
        Imgproc.morphologyEx(mat2, mat2, Imgproc.MORPH_DILATE, kernel);
        displayImageFromMat(mat2);
    }

    private void reverseImage() {
        Core.bitwise_not(mat2, mat2);
        displayImageFromMat(mat2);
    }

    private void doAdaptiveThreshold(Mat m) {
        Mat outerBox = new Mat(m.size(), CV_8UC1);
        Imgproc.adaptiveThreshold(m, outerBox, 250, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 5, 2);
        mat2 = outerBox;
        displayImageFromMat(mat2);

    }

    private void displayImageFromMat(Mat m) {
        Log.i(TAG, "displayImageFromMat: displaying in ImageView");
        // convert to bitmap:
        Bitmap bm = Bitmap.createBitmap(m.cols(), m.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(m, bm);

        // find the imageview and draw it!
        ImageView imageView = (ImageView)findViewById(R.id.imageView);
        imageView.setImageBitmap(bm);
    }


    /*
    attempts to build the outer grid of teh Code cracker grid
     */
    private void doGaussianBlur (Mat m) {
        org.opencv.core.Size s = new Size(17, 17);
        Imgproc.GaussianBlur(mat2, mat2, s, 0);
        displayImageFromMat(mat2);
    }

    public void loadImagefromGallery(View view) {
        int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);

        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }


        // Create intent to Open Image applications like Gallery, Google Photos
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        // Start the Intent
        startActivityForResult(galleryIntent, RESULT_LOAD_IMG);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        try {
            // When an Image is picked
            if (requestCode == RESULT_LOAD_IMG && resultCode == RESULT_OK && null != data) {

                // Get the Image from data
                Uri selectedImage = data.getData();
                String[] filePathColumn = { MediaStore.Images.Media.DATA };

                // Get the cursor
                Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);

                // Move to first row
                cursor.moveToFirst();

                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgDecodableString = cursor.getString(columnIndex);
                cursor.close();
                ImageView imgView = findViewById(R.id.imageView);

                // Set the Image in ImageView after decoding the String
                imgView.setImageBitmap(BitmapFactory.decodeFile(imgDecodableString));
                Utils.bitmapToMat(BitmapFactory.decodeFile(imgDecodableString) , mat2);

            } else {
                Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
        }
    }


    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
}
