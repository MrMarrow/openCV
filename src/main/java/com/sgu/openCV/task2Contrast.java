package com.sgu.openCV;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import static org.opencv.imgproc.Imgproc.COLOR_Lab2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2Lab;

public class task2Contrast {

    public static int L_INDEX = 0;

    public static void main(String[] args) {
        System.load("/C:/Users/Lunge/opencv/build/java/x64/opencv_java420.dll");

        Mat rgbInput = Imgcodecs.imread("inputTask2.png");
        Mat labProcessing = new Mat();

        Imgproc.cvtColor(rgbInput, labProcessing, COLOR_RGB2Lab);
        int rows = labProcessing.rows();
        int cols = labProcessing.cols();

        double maxL = findMaxL(labProcessing);
        double minL = findMinL(labProcessing);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                double[] data = labProcessing.get(i, j);
                data[L_INDEX] = (data[L_INDEX] - minL) / (maxL - minL) * 255;
                labProcessing.put(i, j, data);
            }
        }

        Mat rgbOutpet = new Mat();
        Imgproc.cvtColor(labProcessing, rgbOutpet, COLOR_Lab2RGB);
        Imgcodecs.imwrite("OutputTask2.jpg", rgbOutpet);
    }

    public static double findMaxL(Mat mat) {
        double maxL = mat.get(0, 0)[L_INDEX];
        for (int i = 1; i < mat.rows(); i++) {
            for (int j = 1; j < mat.cols(); j++) {
                double currL = mat.get(i, j)[L_INDEX];
                if (currL > maxL) {
                    maxL = currL;
                }
            }
        }
        return maxL;
    }

    public static double findMinL(Mat mat) {
        double minL = mat.get(0, 0)[L_INDEX];
        for (int i = 1; i < mat.rows(); i++) {
            for (int j = 1; j < mat.cols(); j++) {
                double currL = mat.get(i, j)[L_INDEX];
                if (currL < minL) {
                    minL = currL;
                }
            }
        }
        return minL;
    }
}
