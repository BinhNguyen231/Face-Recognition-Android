package com.binhnn.project2_facerecognition;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Process {

    private static Mat processingMat;
    private static float scale;

    static {
        System.loadLibrary("native-lib");
    }
    public static Mat processingImage(Mat inputMat){
        processingMat = inputMat;
        Mat resizedMat = new Mat();
        scale = 320.f/Math.max(inputMat.width(), inputMat.height());
        Size sz = new Size((int)(scale*inputMat.width()),(int)(scale*inputMat.height()));
        Imgproc.resize(inputMat, resizedMat, sz);
        int[] result = detectFace(resizedMat.getNativeObjAddr());
        if (result== null)
            return null;
        else
            processingMat = displayRecognition(result);
        return processingMat;
    }

    public native static int[] detectFace(long addrInput);

    public static Mat displayRecognition(int[] result){

        Bitmap bmp = Bitmap.createBitmap(processingMat.cols(), processingMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(processingMat, bmp);
        Canvas canvas = new Canvas(bmp);
        Paint mPaint = new Paint();
        mPaint.setColor(Color.GREEN);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setTextSize(50);
        mPaint.setStrokeWidth(10.0f);

        for(int i = 0; i<result.length/5; i++) {
            int label = result[i*5+4];
            int startX = (int)(result[i*5+0] / scale);
            int startY = (int)(result[i*5+1]/scale);
            int endX = (int)(result[i*5+2]/scale);
            int endY = (int)(result[i*5+3]/scale);
            canvas.drawRect(startX, startY, endX, endY, mPaint);
            if(label == 1)
                canvas.drawText("Binh", startX, startY-30, mPaint);
            else if(label == 0)
                canvas.drawText("Unknown", startX, startY-30, mPaint);
        }
        Utils.bitmapToMat(bmp, processingMat);
        return processingMat;
    }
}
